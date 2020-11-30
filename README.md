# Github Pages for Pytorch Text

This branch holds the rendered html for Pytorch Text. The pages are served via
github pages at https://pytorch.org/text/, via a undocumented feature of
github: if one repo in the org has a [CNAME
file](https://github.com/pytorch/pytorch.github.io/blob/site/CNAME) (in this
case [pytorch/pytorch.github.io](https://github.com/pytorch/pytorch.github.io)),
then any other repo in that organization that turns on github pages will be
served under the same CNAME. The branch directory structure reflects the
release history of the project:
- each numbered directory holds the version of the documents at the time of
  release
- there are two special directories: `master` and `stable`. The first holds the
  current HEAD version of the documentation, and can be updated from time to
  time (a future enhancement: merging a PR updates this automatically) via
  something like
  ```
  # get a clean environment
  git checkout master
  git clean -xfd
  virtualenv /tmp/venv
  source /tmp/venv/bin/activate

  # install pytorch text, hopefully this works cleanly
  pip install .

  # build the docs
  pushd docs
  pip install -r requirements.txt
  make html
  popd

  # now docs are built and are in build/html
  # copy them over the current pages
  git checkout gh-pages
  rm -rf master/*
  cp -r docs/build/html/* master
  # let git do its thing
  git add master
  git commit -m"generate new docs"
  git push -u origin
  ```
  The `stable` directory is a symlink to the latest released version, and can
  be recreated via (on a linux machine, does not work via the web GUI nor on
  windows)
  ```
  git checkout gh-pages
  rm stable
  ln -s 0.8 stable   # substitute the correct version number here
  git commit -m "update stable to 0.8"
  git push -u upstream 
  ```
- There is a simple top-level index.html that redirects to `stable/index.html`
  This is needed for naive links to https://pytorch.org/text. Any search
  engine or external links should point to
  https://pytorch.org/text/stable/index.html so they will not need the
  redirect.

