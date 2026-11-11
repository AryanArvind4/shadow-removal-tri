"""Copy one earlier TRI round-ball input for an additional halo check."""
import json
from pathlib import Path
import shutil

from prepare_study import digest


def main():
    study = Path(__file__).resolve().parent
    name = 'Component_G2_1_1152_2472_2140_165_z0060.bmp'
    source = study.parent/'competitor_target_pipeline/G2G3'/name
    destination = study/'round_ball_review'/name
    manifest = study/'study_manifest.json'
    data = json.loads(manifest.read_text())
    if destination.exists() or 'round_ball_review' in data['datasets']:
        raise FileExistsError('Round-ball review is already prepared')
    destination.parent.mkdir()
    shutil.copy2(source, destination)
    if digest(source) != digest(destination):
        raise RuntimeError('Copy verification failed')
    relative = str(destination.relative_to(study))
    data['datasets']['round_ball_review'] = [relative]
    data['inputs'][relative] = digest(source)
    data['additional_review'] = dict(source=str(source), reason='Round-ball halo example; exact screenshot correspondence is not confirmed.')
    manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
