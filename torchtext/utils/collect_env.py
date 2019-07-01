# This script outputs relevant system environment info
# Run it with `python collect_env.py`.
from __future__ import absolute_import, division, print_function, unicode_literals
import re
import subprocess
import sys
import os
from collections import namedtuple

try:
    import torch
    import torchtext
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    TORCH_AVAILABLE = False

PY3 = sys.version_info >= (3, 0)

# System Environment Information
SystemEnv = namedtuple('SystemEnv', [
    'torch_version',
    'torchtext_version',
    'is_debug_build',
    'cuda_compiled_version',
    'gcc_version',
    'cmake_version',
    'os',
    'python_version',
    'is_cuda_available',
    'cuda_runtime_version',
    'nvidia_driver_version',
    'nvidia_gpu_models',
    'cudnn_version',
    'pip_version',  # 'pip' or 'pip3'
    'pip_packages',
    'conda_packages',
])


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        output = output.decode("utf-8")
        err = err.decode("utf-8")
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_conda_packages(run_lambda):
    if get_platform() == 'win32':
        grep_cmd = r'findstr /R "torch soumith mkl magma"'
    else:
        grep_cmd = r'grep "torch\|soumith\|mkl\|magma"'
    conda = os.environ.get('CONDA_EXE', 'conda')
    out = run_and_read_all(run_lambda, conda + ' list | ' + grep_cmd)
    if out is None:
        return out
    # Comment starting at beginning of line
    comment_regex = re.compile(r'^#.*\n')
    return re.sub(comment_regex, '', out)


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'gcc --version', r'gcc (.*)')


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version', r'cmake (.*)')


def get_nvidia_driver_version(run_lambda):
    if get_platform() == 'darwin':
        cmd = 'kextstat | grep -i cuda'
        return run_and_parse_first_match(run_lambda, cmd,
                                         r'com[.]nvidia[.]CUDA [(](.*?)[)]')
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r'Driver Version: (.*?) ')


def get_gpu_info(run_lambda):
    if get_platform() == 'darwin':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r' \(UUID: .+?\)')
    rc, out, _ = run_lambda(smi + ' -L')
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, '', out)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', r'V(.*)$')


def get_cudnn_version(run_lambda):
    """This will return a list of libcudnn"""
    if get_platform() == 'win32':
        cudnn_cmd = 'where /R "%CUDA_PATH%\\bin" cudnn*.dll'
    elif get_platform() == 'darwin':
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#install
        # https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = 'ls /usr/local/cuda/lib/libcudnn*'
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        lib = os.environ.get('CUDNN_LIBRARY')
        if lib is not None and os.path.isfile(lib):
            return os.path.realpath(lib)
        return None
    files = set()
    for fn in out.split('\n'):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files.add(fn)
    if not files:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = list(sorted(files))
    if len(files) == 1:
        return files[0]
    result = '\n'.join(files)
    return 'Probably one of the following:\n{}'.format(result)


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = 'nvidia-smi'
    if get_platform() == 'win32':
        smi = '"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\%s"' % smi
    return smi


def get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('win32'):
        return 'win32'
    elif sys.platform.startswith('cygwin'):
        return 'cygwin'
    elif sys.platform.startswith('darwin'):
        return 'darwin'
    else:
        return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'sw_vers -productVersion', r'(.*)')


def get_windows_version(run_lambda):
    return run_and_read_all(run_lambda, 'wmic os get Caption | findstr /v Caption')


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'lsb_release -a', r'Description:\t(.*)')


def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cat /etc/*-release',
                                     r'PRETTY_NAME="(.*)"')


def get_os(run_lambda):
    platform = get_platform()

    if platform == 'win32' or platform == 'cygwin':
        return get_windows_version(run_lambda)

    if platform == 'darwin':
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return 'Mac OSX {}'.format(version)

    if platform == 'linux':
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return desc

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return desc

        return platform

    # Unknown platform
    return platform


def get_pip_packages(run_lambda):
    # People generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        if get_platform() == 'win32':
            grep_cmd = r'findstr /R "numpy torch"'
        else:
            grep_cmd = r'grep "torch\|numpy"'
        return run_and_read_all(run_lambda, pip + ' list --format=freeze | ' + grep_cmd)

    if not PY3:
        return 'pip', run_with_pip('pip')

    # Try to figure out if the user is running pip or pip3.
    out2 = run_with_pip('pip')
    out3 = run_with_pip('pip3')

    num_pips = len([x for x in [out2, out3] if x is not None])
    if num_pips == 0:
        return 'pip', out2

    if num_pips == 1:
        if out2 is not None:
            return 'pip', out2
        return 'pip3', out3

    # num_pips is 2. Return pip3 by default b/c that most likely
    # is the one associated with Python 3
    return 'pip3', out3


def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        torchtext_version_str = torchtext.__version__
        debug_mode_str = torch.version.debug
        cuda_available_str = torch.cuda.is_available()
        cuda_version_str = torch.version.cuda
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = 'N/A'
        torchtext_version_str = 'N/A'

    return SystemEnv(
        torch_version=version_str,
        torchtext_version=torchtext_version_str,
        is_debug_build=debug_mode_str,
        python_version='{}.{}'.format(sys.version_info[0], sys.version_info[1]),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=get_conda_packages(run_lambda),
        os=get_os(run_lambda),
        gcc_version=get_gcc_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
TorchText version: {torchtext_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}

OS: {os}
GCC version: {gcc_version}
CMake version: {cmake_version}

Python version: {python_version}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict['nvidia_gpu_models'] = \
        maybe_start_on_next_line(envinfo.nvidia_gpu_models)

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        'cuda_runtime_version',
        'nvidia_gpu_models',
        'nvidia_driver_version',
    ]
    all_cuda_fields = dynamic_cuda_fields + ['cudnn_version']
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields)
    if TORCH_AVAILABLE and not torch.cuda.is_available() \
            and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = 'No CUDA'
        if envinfo.cuda_compiled_version is None:
            mutable_dict['cuda_compiled_version'] = 'None'

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict['pip_packages'] = replace_if_empty(mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(mutable_dict['conda_packages'])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict['pip_packages']:
        mutable_dict['pip_packages'] = prepend(mutable_dict['pip_packages'],
                                               '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        mutable_dict['conda_packages'] = prepend(mutable_dict['conda_packages'],
                                                 '[conda] ')
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)


if __name__ == '__main__':
    main()
