#!/bin/bash
package_type="$PACKAGE_TYPE"
channel="$CHANNEl"
if [ -z "$package_type" ]; then
  package_type="wheel"
fi
if [ -z "$channel" ]; then
  channel="nightly"
fi

# Wrong values
if [ "$package_type" != "wheel" ] && [ "$package_type" != "conda" ]; then
  exit 1
fi
if [ "$channel" != "nightly" ] && [ "$channel" != "test" ]; then
  exit 1
fi


if [ "$package_type" = "wheel" ]; then
  install_cmd="pip install"
  if [ "$channel" = "nightly" ]; then
    install_cmd="${install_cmd} --pre"
    install_channel="--extra-index-url https://download.pytorch.org/whl/nightly/cpu"
  else
    install_channel="--extra-index-url https://download.pytorch.org/whl/test/cpu"
  fi
else
  install_cmd="conda install"
  if [ "$channel" = "nightly" ]; then
    install_channel="-c pytorch-nightly"
  else
    install_channel="-c pytorch-test"
  fi
fi

$install_cmd torchdata $install_channel

if [ "$package_type" = "wheel" ]; then
  TORCHDATA_VERSION="$(pip show torchdata | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
else
  TORCHDATA_VERSION="$(conda search --json torchdata[channel=pytorch-${channel}] | \
	  python -c "import json, os, re, sys; \
	    cuver = 'cpu'; \
	    pyver = os.environ.get('PYTHON_VERSION'); \
	    print(re.sub(r'\\+.*$', '',
	      [x['version'] for x in json.load(sys.stdin)['torchdata'] \
		      if 'py' + pyver in x['fn']][-1]))"
	    )"
fi

echo "export TORCHDATA_VERSION=${TORCHDATA_VERSION}" >> ${BUILD_ENV_FILE}
