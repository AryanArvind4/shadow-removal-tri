"""Prepare, annotate, and evaluate a small human-reviewed PCB image set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import competitor_target_v3 as v3
import competitor_target_v4 as v4
import competitor_target_v5 as v5
from evaluate_reviewed_regions import polygon_mask, measure

REVIEW = Path(__file__).resolve().parent / 'review'
MANIFEST = REVIEW / 'annotations.json'
SELECTED = ['J1_000.jpg','J1_007.jpg','J4_044.jpg','J4_049.jpg',
            'U29_030.jpg','U29_034.jpg','U30_081.jpg','U30_085.jpg']
KINDS = ('healthy','defect','background','pad_boundary')
COLORS = dict(healthy='#36d985',defect='#ff635e',background='#45afff',pad_boundary='#ffda55')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def validate_entry(entry, source, require_review=True):
    if Path(entry['filename']).name != entry['filename']:
        raise ValueError('Image entries must use filenames only')
    if digest(source)!=entry['sha256']:
        raise ValueError(f'Source changed: {entry["filename"]}')
    with Image.open(source) as image:
        shape=(image.height,image.width)
    if list(shape)!=entry['shape']:
        raise ValueError('Source dimensions changed')
    if require_review and (not entry.get('reviewed') or not entry.get('reviewer','').strip()):
        raise ValueError('Human review and reviewer name are required')
    masks=[]
    for item in entry['annotations']:
        if item['kind'] not in KINDS or not str(item['pad_id']).strip():
            raise ValueError('Invalid annotation category or pad ID')
        mask=polygon_mask(item['points'],shape)
        for prior,prior_mask in masks:
            if {prior['kind'],item['kind']}=={'healthy','defect'} and np.any(prior_mask&mask):
                raise ValueError('Healthy and defect polygons overlap')
            if 'background' in {prior['kind'],item['kind']} and prior['kind']!=item['kind'] and np.any(prior_mask&mask):
                raise ValueError('Background overlaps a solder/defect annotation')
        masks.append((item,mask))
    if require_review and not masks:
        raise ValueError('No annotations to review')
    if require_review:
        kinds={item['kind'] for item,_ in masks}
        if not {'healthy','background'}.issubset(kinds):
            raise ValueError('Mark at least one healthy solder region and one background region.')
        for item,_ in masks:
            if item['kind']=='defect' and not any(prior['kind']=='healthy' and str(prior['pad_id'])==str(item['pad_id']) for prior,_ in masks):
                raise ValueError('Each defect needs a healthy reference polygon with the same pad ID.')
    return masks


def make_crops(entry, destination):
    raw=v3.read(v3.INPUT/entry['filename'])
    images=[('Input',raw)]
    for name,root in [('V4',v4.OUTPUT),('V5 automatic',v5.AUTO_OUTPUT)]:
        path=root/'reference_gray'/(Path(entry['filename']).stem+'.png')
        if path.exists(): images.append((name,v3.read(path)))
    h,w=raw.shape
    boxes=[('upper_left',(.04,.06,.43,.26)),('middle_left',(.04,.3,.44,.72)),('middle_right',(.54,.3,.94,.72))]
    for label,(a,b,c,d) in boxes:
        x0,y0,x1,y1=int(a*w),int(b*h),int(c*w),int(d*h)
        width,height=x1-x0,y1-y0
        sheet=Image.new('RGB',(width*len(images),height+30),(24,24,24))
        draw=ImageDraw.Draw(sheet)
        for index,(name,image) in enumerate(images):
            sheet.paste(Image.fromarray(v3.u8(image[y0:y1,x0:x1])).convert('RGB'),(index*width,30))
            draw.text((index*width+5,6),name,fill='white')
        sheet.save(destination/(Path(entry['filename']).stem+'_'+label+'.png'))
    return [dict(name=name,relative_box=list(box)) for name,box in boxes]


def prepare(path):
    data=json.loads(path.read_text()) if path.exists() else dict(schema=1,input=str(v3.INPUT),
        note='Unreviewed candidates. No automatic defect or healthy labels. Previously viewed images; not an unseen holdout.',images=[])
    known={e['filename']:e for e in data['images']}
    for filename in SELECTED:
        source=v3.INPUT/filename
        if filename not in known:
            with Image.open(source) as image:
                entry=dict(filename=filename,shape=[image.height,image.width],sha256=digest(source),
                           reviewed=False,reviewer='',annotations=[])
            data['images'].append(entry)
        else:
            validate_entry(known[filename],source,require_review=False)
    crops=path.parent/'matched_crops'
    crops.mkdir(parents=True,exist_ok=True)
    for entry in data['images']:
        entry['crop_windows']=make_crops(entry,crops)
    save(path,data)
    # Export metadata is evidence about the available files, not detector settings.
    audit=[]
    for source in sorted(v3.INPUT.glob('U30*')):
        with Image.open(source) as image:
            exif=image.getexif()
            audit.append(dict(filename=source.name,format=image.format,mode=image.mode,
                              size=list(image.size),sha256=digest(source),
                              description=str(exif.get(270,'')),software=str(exif.get(305,'')),
                              exif_tags=len(exif),export_info_keys=list(image.info)))
    dataset=v3.INPUT.parent
    sidecars=[str(p.relative_to(dataset)) for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.json','.xml','.csv','.txt','.yaml','.yml','.dcm','.raw'}]
    save(path.parent/'u30_source_audit.json',dict(images=audit,sidecar_candidates=sidecars,
         open_questions=['Are these 2D projections or reconstructed slices?', 'Do source and reference show the same specimen and depth?',
                         'Can TRI supply native bit depth, pixel size, projection data or slice stacks, and an unprocessed matching competitor input?'],
         conclusion='Image exports alone do not establish acquisition mode, depth, or missing cavity information.'))
    print(f'Review manifest: {path.resolve()}')
    print('Labels are pending human review. Existing annotations have been retained.')


def evaluate(path):
    data=json.loads(path.read_text())
    rows=[]
    reviewed=0
    for entry in data['images']:
        if not entry.get('reviewed'): continue
        source=Path(data['input'])/entry['filename']
        masks=validate_entry(entry,source)
        reviewed+=1
        images={'input':v3.read(source)}
        for name,root in [('v4',v4.OUTPUT),('v5',v5.AUTO_OUTPUT)]:
            images[name]=v3.read(root/'reference_gray'/(Path(entry['filename']).stem+'.png'))
            if list(images[name].shape)!=entry['shape']: raise ValueError('Output dimensions differ')
        for pad in sorted({str(item['pad_id']) for item,_ in masks}):
            selected=[(item,mask) for item,mask in masks if str(item['pad_id'])==pad]
            merged={kind:np.logical_or.reduce([mask for item,mask in selected if item['kind']==kind]) for kind in KINDS if any(item['kind']==kind for item,_ in selected)}
            for method,image in images.items():
                record=dict(filename=entry['filename'],pad_id=pad,method=method,reviewer=entry['reviewer'],
                            healthy_std=None,healthy_mean=None,defect_mean=None,signed_healthy_minus_defect=None,cnr=None,
                            background_std=None,background_high_frequency_rms=None)
                if 'healthy' in merged:
                    values=image[merged['healthy']]
                    record.update(healthy_std=float(values.std()),healthy_mean=float(values.mean()))
                if 'healthy' in merged and 'defect' in merged:
                    measured=measure(image,merged['healthy'],merged['defect'])
                    record.update({key:measured[key] for key in ['defect_mean','signed_healthy_minus_defect','cnr']})
                if 'background' in merged:
                    mask=merged['background']
                    residual=image-v3.blur(image,1.)
                    record.update(background_std=float(image[mask].std()),background_high_frequency_rms=float(np.sqrt(np.mean(residual[mask]**2))))
                rows.append(record)
    if not reviewed:
        print('No human-reviewed entries. No reviewed metrics were generated.')
        return
    v3.write_csv(path.parent/'reviewed_metrics.csv',rows)
    save(path.parent/'evaluation_provenance.json',dict(annotation_sha256=digest(path),reviewed_images=reviewed,
        pending_images=len(data['images'])-reviewed,note='Fixed-polygon contrast only; defect boundary/area preservation is not validated.'))
    print(f'Evaluated {reviewed} reviewed images. Metrics: {path.parent / "reviewed_metrics.csv"}')


class Workbench:
    def __init__(self,root,path):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk,self.messagebox=tk,messagebox
        self.root,self.path=root,path
        self.data=json.loads(path.read_text())
        self.index,self.scale,self.points=0,1.,[]
        self.kind=tk.StringVar(value='healthy')
        self.pad=tk.StringVar(value='1')
        self.reviewer=tk.StringVar()
        self.view=tk.StringVar(value='Input')
        self.status=tk.StringVar()
        root.title('TRI region review')
        root.geometry('1180x900')
        bar=ttk.Frame(root);bar.pack(fill='x')
        for text,command in [('Previous',lambda:self.navigate(-1)),('Next',lambda:self.navigate(1)),('Fit',self.fit),('+',lambda:self.zoom(1.3)),('-',lambda:self.zoom(1/1.3))]:
            ttk.Button(bar,text=text,command=command).pack(side='left')
        menu=ttk.Combobox(bar,textvariable=self.view,values=['Input','V4','V5'],state='readonly',width=8)
        menu.pack(side='left');menu.bind('<<ComboboxSelected>>',lambda e:self.draw())
        ttk.Label(bar,text='Type').pack(side='left')
        ttk.Combobox(bar,textvariable=self.kind,values=KINDS,state='readonly',width=14).pack(side='left')
        ttk.Label(bar,text='Pad ID').pack(side='left')
        ttk.Entry(bar,textvariable=self.pad,width=7).pack(side='left')
        controls=ttk.Frame(root);controls.pack(fill='x')
        for text,command in [('Undo point',self.undo),('Finish polygon',self.finish),('Delete selected',self.delete),('Save draft',self.store),('Mark image reviewed',self.approve)]:
            ttk.Button(controls,text=text,command=command).pack(side='left')
        ttk.Label(controls,text='Reviewer').pack(side='left')
        ttk.Entry(controls,textvariable=self.reviewer,width=18).pack(side='left')
        frame=ttk.Frame(root);frame.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(frame,bg='#202020',highlightthickness=0)
        xs=ttk.Scrollbar(frame,orient='horizontal',command=self.canvas.xview)
        ys=ttk.Scrollbar(frame,orient='vertical',command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xs.set,yscrollcommand=ys.set)
        self.canvas.grid(row=0,column=0,sticky='nsew');ys.grid(row=0,column=1,sticky='ns');xs.grid(row=1,column=0,sticky='ew')
        frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
        self.listbox=tk.Listbox(frame,width=28);self.listbox.grid(row=0,column=2,sticky='ns')
        self.canvas.bind('<Button-1>',self.add_point)
        self.canvas.bind('<Button-3>',lambda e:self.finish())
        ttk.Label(root,textvariable=self.status).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.update_idletasks();self.fit()

    @property
    def entry(self): return self.data['images'][self.index]

    def store(self): save(self.path,self.data)
    def close(self): self.store();self.root.destroy()
    def navigate(self,delta):
        self.store();self.points=[];self.index=(self.index+delta)%len(self.data['images']);self.fit()
    def fit(self):
        h,w=self.entry['shape'];self.scale=min(max(200,self.canvas.winfo_width())/w,max(200,self.canvas.winfo_height())/h,1.)
        self.draw()
    def zoom(self,factor): self.scale=np.clip(self.scale*factor,.1,5.);self.draw()
    def draw(self):
        filename=self.entry['filename']
        source=Path(self.data['input'])/filename
        if self.view.get()!='Input':
            source=(v4.OUTPUT if self.view.get()=='V4' else v5.AUTO_OUTPUT)/'reference_gray'/(Path(filename).stem+'.png')
        if not source.exists():
            self.messagebox.showerror('Missing output',str(source));self.view.set('Input');source=Path(self.data['input'])/filename
        with Image.open(source) as image:
            self.photo=ImageTk.PhotoImage(image.convert('RGB').resize((round(image.width*self.scale),round(image.height*self.scale)),Image.Resampling.NEAREST))
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.canvas.configure(scrollregion=(0,0,self.photo.width(),self.photo.height()))
        self.listbox.delete(0,'end')
        for item in self.entry['annotations']:
            coords=[coordinate*self.scale for point in item['points'] for coordinate in point]
            self.canvas.create_polygon(*coords,outline=COLORS[item['kind']],fill='',width=2)
            self.listbox.insert('end',f'{item["kind"]}: pad {item["pad_id"]}')
        if self.points:
            coords=[coordinate*self.scale for point in self.points for coordinate in point]
            if len(self.points)>1: self.canvas.create_line(*coords,fill=COLORS[self.kind.get()],width=2)
            for x,y in self.points:
                x,y=x*self.scale,y*self.scale
                self.canvas.create_oval(x-3,y-3,x+3,y+3,fill=COLORS[self.kind.get()])
        self.status.set(f'{filename} | {self.index+1}/{len(self.data["images"])} | {"Reviewed" if self.entry["reviewed"] else "Pending review"} | Zoom {self.scale:.2f}')
    def add_point(self,event):
        x,y=self.canvas.canvasx(event.x)/self.scale,self.canvas.canvasy(event.y)/self.scale
        h,w=self.entry['shape']
        if 0<=x<w and 0<=y<h: self.points.append([round(min(x,w-1),2),round(min(y,h-1),2)]);self.draw()
    def undo(self):
        if self.points:self.points.pop();self.draw()
    def finish(self):
        item=dict(kind=self.kind.get(),pad_id=self.pad.get().strip(),points=self.points.copy())
        candidate=dict(self.entry,annotations=self.entry['annotations']+[item])
        try: validate_entry(candidate,Path(self.data['input'])/self.entry['filename'],False)
        except ValueError as error:self.messagebox.showerror('Invalid polygon',str(error));return
        self.entry['annotations'].append(item);self.entry['reviewed']=False;self.points=[];self.store();self.draw()
    def delete(self):
        selected=self.listbox.curselection()
        if selected: del self.entry['annotations'][selected[0]];self.entry['reviewed']=False;self.store();self.draw()
    def approve(self):
        candidate=dict(self.entry,reviewed=True,reviewer=self.reviewer.get().strip())
        try:validate_entry(candidate,Path(self.data['input'])/self.entry['filename'])
        except ValueError as error:self.messagebox.showerror('Review incomplete',str(error));return
        if self.points:self.messagebox.showerror('Unfinished polygon','Finish or undo the current polygon first.');return
        self.entry.update(reviewed=True,reviewer=self.reviewer.get().strip());self.store();self.draw()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','annotate','evaluate'])
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.manifest)
    elif args.action=='evaluate':evaluate(args.manifest)
    else:
        if not args.manifest.exists():prepare(args.manifest)
        import tkinter as tk
        root=tk.Tk();Workbench(root,args.manifest);root.mainloop()


if __name__=='__main__':main()
