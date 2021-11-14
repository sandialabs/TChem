# Generating README.md

For the convenience of organization and editing, we do documentation separately and harness them into a single ``README.md`` using ``markdown-pp``. This illustrates how to use ``markdown-pp``.

## Installing Pip

``markdown-pp`` can be installed through ``pip``. If ``pip`` is not installed on your local machine, it can be obtained as follows.

  0. Confirm that you have a version of python on your machine.
  1. Download ``get-pip.py`` from [here](https://bootstrap.pypa.io/get-pip.py).
  2. Run ``python get-pip.py``. Then, check its availability.
  ```
> which pip
pip is /Users/bob/Library/Python/2.7/bin/pip
  ```
  3. Using a Mac system, a user may need to run the following command.
  ```
> sudo python -m pip install -U pip==8.0.1
You are using pip version 7.1.0, however version 20.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
Collecting pip==8.0.1
  Downloading https://files.pythonhosted.org/packages/45/9c/6f9a24917c860873e2ce7bd95b8f79897524353df51d5d920cd6b6c1ec33/pip-8.0.1-py2.py3-none-any.whl (1.2MB)
    100% |████████████████████████████████| 1.2MB 380kB/s 
Installing collected packages: pip
Found existing installation: pip 7.1.0
Uninstalling pip-7.1.0:
Successfully uninstalled pip-7.1.0
Successfully installed pip-20.1
  ```

## Installing Markdown-pp

  0. Clone the repository at ``git clone https://github.com/jreese/markdown-pp.git /your/markdown-pp``.
  1. Run ``pip install MarkdownPP``.
  ```
Collecting MarkdownPP
  Downloading MarkdownPP-1.5.1.tar.gz (14 kB)
Collecting Watchdog>=0.8.3
  Downloading watchdog-0.10.2.tar.gz (95 kB)
     |████████████████████████████████| 95 kB 892 kB/s 
Collecting pathtools>=0.1.1
  Downloading pathtools-0.1.2.tar.gz (11 kB)
Building wheels for collected packages: MarkdownPP, Watchdog, pathtools
...
Successfully built MarkdownPP Watchdog pathtools
Installing collected packages: pathtools, Watchdog, MarkdownPP
Successfully installed MarkdownPP-1.5.1 Watchdog-0.10.2 pathtools-0.1.2
  ```  
  2. Check its availability.
  ```
> which markdown-pp 
markdown-pp is /Users/bob/Library/Python/2.7/bin/markdown-pp   
  ```

## Harnessing md files by running the following command. 

```
markdown-pp -e latexrender front.mdpp -o front.md
```

Check the output. If the document is okay, copy it to the front README.md in the top-level directory.

# Clang-Format

We maintain our consistent coding style using ``clang-format``. The coding style is defined in ``src/.clang-format``. The usage is simple as follows.

```
/// any version of clang-format should work. this clang format happens to be on my laptop
clang-format-mp-9.0 -i --style=file  example/*.*pp unit-test/*.*pp
clang-format-mp-9.0 -i --style=file  core/*.*pp core/problems/*.*pp core/impl/*.*pp
```
