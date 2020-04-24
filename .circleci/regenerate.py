#!/usr/bin/env python3

"""
This script should use a very simple, functional programming style.
Avoid Jinja macros in favor of native Python functions.

Don't go overboard on code generation; use Python only to generate
content that can't be easily declared statically using CircleCI's YAML API.

Data declarations (e.g. the nested loops for defining the configuration matrix)
should be at the top of the file for easy updating.

See this comment for design rationale:
https://github.com/pytorch/vision/pull/1321#issuecomment-531033978
"""

import jinja2
import yaml
import os.path


def workflows(prefix='', upload=False, filter_branch=None, indentation=6):
    w = []
    for btype in ["wheel", "conda"]:
        for os_type in ["linux", "macos"]:
            for python_version in ["3.6", "3.7", "3.8"]:
                w += workflow_pair(btype, os_type, python_version, filter_branch, prefix, upload)

    return indent(indentation, w)


def workflow_pair(btype, os_type, python_version, filter_branch, prefix='', upload=False):

    w = []
    base_workflow_name = "{prefix}binary_{os_type}_{btype}_py{python_version}".format(
        prefix=prefix,
        os_type=os_type,
        btype=btype,
        python_version=python_version,
    )

    w.append(generate_base_workflow(base_workflow_name, python_version, filter_branch, os_type, btype))

    if upload:

        is_py3_linux = os_type == 'linux' and not python_version.startswith("2.")

        w.append(generate_upload_workflow(base_workflow_name, filter_branch, btype))

        if filter_branch == 'nightly' and is_py3_linux:
            pydistro = 'pip' if btype == 'wheel' else 'conda'
            w.append(generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version))

    return w


def generate_base_workflow(base_workflow_name, python_version, filter_branch, os_type, btype):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {"binary_{os_type}_{btype}".format(os_type=os_type, btype=btype): d}


def gen_filter_branch_tree(branch_name):
    return {"branches": {"only": branch_name}}


def generate_upload_workflow(base_workflow_name, filter_branch, btype):
    d = {
        "name": "{base_workflow_name}_upload".format(base_workflow_name=base_workflow_name),
        "context": "org-member",
        "requires": [base_workflow_name],
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {"binary_{btype}_upload".format(btype=btype): d}


def generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version):

    required_build_suffix = "_upload"
    required_build_name = base_workflow_name + required_build_suffix

    smoke_suffix = "smoke_test_{pydistro}".format(pydistro=pydistro)
    d = {
        "name": "{base_workflow_name}_{smoke_suffix}".format(
            base_workflow_name=base_workflow_name, smoke_suffix=smoke_suffix),
        "requires": [required_build_name],
        "python_version": python_version,
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {"smoke_test_linux_{pydistro}".format(pydistro=pydistro): d}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(yaml.dump(data_list).splitlines())


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=False,
    )

    with open(os.path.join(d, 'config.yml'), 'w') as f:
        f.write(env.get_template('config.yml.in').render(workflows=workflows))
