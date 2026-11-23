"""Compile the frozen Cython/C++ source without downloading or vendoring tools."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import numpy as np
from Cython.Build import cythonize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default='g++', help='C++ compiler executable')
    args = parser.parse_args()
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise RuntimeError('A C++11 compiler is required; no automatic installation')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    cythonize(str(ROOT / 'native/NIMF.pyx'), build_dir=str(build), language='c++',
              compiler_directives={'language_level': 3}, force=True, quiet=True)
    generated = list(build.rglob('NIMF.cpp'))
    if len(generated) != 1:
        raise RuntimeError('Expected exactly one generated Cython source')
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.pyd'
    output = ROOT / ('NIMF' + suffix)
    command = [compiler, '-shared', '-O2', '-std=c++11',
               '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
               '-I' + sysconfig.get_path('include'), '-I' + np.get_include(),
               '-I' + str(ROOT / 'native'), str(generated[0]),
               str(ROOT / 'native/c_NIMF.cpp'), str(ROOT / 'native/c_UIPCC.cpp')]
    if sys.platform == 'win32':
        command += ['-DMS_WIN64', '-static-libgcc', '-static-libstdc++',
                    str(Path(sys.base_prefix) / f'python{sys.version_info.major}{sys.version_info.minor}.dll')]
    else:
        command += ['-fPIC']
        if sys.platform == 'darwin':
            command += ['-undefined', 'dynamic_lookup']
    command += ['-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    (build / 'compiler.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    (build / 'build.json').write_text(json.dumps(dict(returncode=result.returncode,
        python=sys.version, numpy=np.__version__, compiler=Path(compiler).name,
        flags='-O2 -std=c++11; no fast-math'), indent=2), encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Native compilation failed; see build/compiler.log')
    if sys.platform == 'win32':
        runtime = Path(compiler).parent / 'libwinpthread-1.dll'
        if runtime.exists():
            shutil.copyfile(runtime, ROOT / runtime.name)
    print('Built', output.name)


if __name__ == '__main__':
    main()
