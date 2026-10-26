"""Bind a local scene package to independent UE workspaces; never launch UE or upload assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile


DEFAULT_SCENE = 'mondstadt-city-7-1-0'
PROJECT = 'UE5/MondstadtCityDetailed.uproject'
MAP = 'UE5/Content/MondstadtCity/Mondstadt_City_Detailed.umap'
MAP_PATH = '/Game/MondstadtCity/Mondstadt_City_Detailed'
ASSET_PREFIX = 'UE5/Content/MondstadtCity/'


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def scene_name(value):
    reserved = {'con', 'prn', 'aux', 'nul'} | {f'{p}{i}' for p in ('com', 'lpt') for i in range(1, 10)}
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', value) or value in reserved:
        raise ValueError('scene id must be a portable lowercase slug')
    return value


def safe_relative(value):
    if not isinstance(value, str) or not value or ':' in value:
        raise ValueError('unsafe manifest path')
    value = value.replace('\\', '/')
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('unsafe manifest path')
    return value


def unlinked(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(part)):
            raise ValueError('linked paths are not supported by the scene copier')
    return path


def json_read(path):
    return json.loads(unlinked(path).read_text(encoding='utf-8-sig'))


def json_write(path, data):
    unlinked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package_inventory(source):
    source = unlinked(Path(os.path.abspath(source)))
    manifest = source / 'delivery_checksums.json'
    entries = json_read(manifest)
    if not isinstance(entries, list):
        raise ValueError('delivery checksums must be a list')
    inventory = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('invalid delivery entry')
        name = safe_relative(entry.get('path'))
        checksum = entry.get('sha256')
        if name in inventory or not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise ValueError('duplicate path or invalid delivery digest')
        if type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            raise ValueError('invalid delivery size')
        inventory[name] = entry
    if PROJECT not in inventory or MAP not in inventory:
        raise ValueError('detailed city project and map must be covered by the delivery manifest')
    files = {name: entry for name, entry in inventory.items() if name == PROJECT or
             name.startswith(ASSET_PREFIX) and Path(name).suffix in ('.uasset', '.umap') or
             name == 'UE5/Config/DefaultInput.ini'}
    actual = {str(file.relative_to(source)).replace('\\', '/')
              for file in (source / ASSET_PREFIX).rglob('*') if file.suffix in ('.uasset', '.umap')}
    if actual != {name for name in files if name.startswith(ASSET_PREFIX)}:
        raise ValueError('asset inventory differs from delivery manifest')
    for name, entry in files.items():
        path = unlinked(source / name)
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('missing or changed scene source file: ' + name)
    project = json_read(source / PROJECT)
    if not isinstance(project, dict) or project.get('EngineAssociation') != '5.8':
        raise ValueError('this local scene profile requires the UE 5.8 detailed city project')
    validation = json_read(source / 'city_ue_saved_validation.json')
    if not isinstance(validation, dict) or validation.get('status') != 'success':
        raise ValueError('source package has no successful saved-map validation report')
    return source, digest(manifest), files


def catalog_read(path):
    catalog = json_read(path)
    if not isinstance(catalog, dict) or catalog.get('schema_version') != 1 or not isinstance(catalog.get('scenes'), dict):
        raise ValueError('invalid local scene catalog')
    return catalog


def register_scene(catalog_path, source, scene_id=DEFAULT_SCENE):
    scene_name(scene_id)
    source, checksum, files = package_inventory(source)
    catalog_path = Path(os.path.abspath(catalog_path))
    catalog = catalog_read(catalog_path) if catalog_path.exists() else {'schema_version': 1, 'scenes': {}}
    entry = {'source_package': str(source), 'delivery_sha256': checksum,
             'profile': 'mondstadt-city-detailed-ue58', 'ue_project': PROJECT, 'map_path': MAP_PATH,
             'asset_files': sum(x.startswith(ASSET_PREFIX) for x in files),
             'physics_navigation_play_tested': False}
    existing = catalog['scenes'].get(scene_id)
    if existing is not None and existing != entry:
        raise ValueError('scene id is already registered differently; use a new versioned id')
    if existing is None:
        catalog['scenes'][scene_id] = entry
        json_write(catalog_path, catalog)
    return entry


def binding_path(repo, scene_id):
    return unlinked(Path(os.path.abspath(repo)) / 'local/scenes' / scene_name(scene_id) / 'binding.local.json')


def prepare_scene(repo, catalog_path, scene_id=DEFAULT_SCENE):
    scene_name(scene_id)
    catalog_path = Path(os.path.abspath(catalog_path))
    entry = catalog_read(catalog_path)['scenes'].get(scene_id)
    if not isinstance(entry, dict):
        raise ValueError('scene is not registered in the selected catalog')
    source, checksum, files = package_inventory(entry.get('source_package', ''))
    if checksum != entry.get('delivery_sha256'):
        raise ValueError('source delivery manifest changed after registration')
    receipt = binding_path(repo, scene_id)
    target = receipt.parent
    if target.exists():
        if not receipt.is_file():
            raise ValueError('refusing to overwrite an existing unregistered workspace')
        previous = json_read(receipt)
        if previous.get('source_package') != str(source) or previous.get('delivery_sha256') != checksum:
            raise ValueError('existing workspace belongs to a different scene source')
        return previous  # Explicitly preserve all experiment edits.
    # Verify the complete source before creating any destination.
    target.mkdir(parents=True)
    workspace = target / 'workspace'
    for name, info in files.items():
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)  # Separate copies, never links or hard links.
        if digest(destination) != info['sha256']:
            raise ValueError('copied workspace failed integrity check: ' + name)
    config = workspace / 'UE5/Config/DefaultEngine.ini'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[/Script/EngineSettings.GameMapsSettings]\n'
                      f'EditorStartupMap={MAP_PATH}\nGameDefaultMap={MAP_PATH}\n',
                      encoding='utf-8', newline='\n')
    copied = {name: info['sha256'] for name, info in files.items()}
    copied['UE5/Config/DefaultEngine.ini'] = digest(config)
    result = {'schema_version': 1, 'scene_id': scene_id, 'catalog': str(catalog_path),
              'source_package': str(source), 'delivery_sha256': checksum,
              'workspace': str(workspace), 'ue_project': str(workspace / PROJECT),
              'map_path': MAP_PATH, 'files': copied, 'physics_navigation_play_tested': False,
              'note': 'Independent static scene copy; navigation, capture, rewards and gameplay require separate integration.'}
    json_write(receipt, result)
    return result


def inspect_scene(repo, scene_id=DEFAULT_SCENE):
    receipt = binding_path(repo, scene_id)
    binding = json_read(receipt)
    expected_workspace = receipt.parent / 'workspace'
    if not isinstance(binding, dict) or binding.get('schema_version') != 1 or binding.get('scene_id') != scene_id:
        raise ValueError('invalid scene binding')
    if binding.get('workspace') != str(expected_workspace) or not isinstance(binding.get('files'), dict):
        raise ValueError('invalid workspace in scene binding')
    missing, changed = [], []
    for name, checksum in binding['files'].items():
        path = unlinked(expected_workspace / safe_relative(name))
        if not path.is_file():
            missing.append(name)
        elif digest(path) != checksum:
            changed.append(name)
    return {'scene_id': scene_id, 'source_package': binding['source_package'],
            'ue_project': binding['ue_project'], 'workspace_files': len(binding['files']),
            'workspace_missing_files': missing, 'workspace_modified_files': changed,
            'physics_navigation_play_tested': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('register', 'prepare', 'inspect'):
        command = commands.add_parser(name)
        command.add_argument('--scene', default=DEFAULT_SCENE)
        if name in ('register', 'prepare'):
            command.add_argument('--catalog', required=True, type=Path)
        if name == 'register':
            command.add_argument('--source-package', required=True, type=Path)
        else:
            command.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        if args.command == 'register':
            result = register_scene(args.catalog, args.source_package, args.scene)
        elif args.command == 'prepare':
            result = prepare_scene(args.root, args.catalog, args.scene)
            result = {k: v for k, v in result.items() if k != 'files'}
        else:
            result = inspect_scene(args.root, args.scene)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 2 if result.get('workspace_missing_files') else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print('Scene binding failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
