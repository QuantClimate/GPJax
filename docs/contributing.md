# Contributing

## How can I contribute?

GPJax welcomes contributions from interested individuals or groups. There are
many ways to contribute, including:

- Answering questions on our [discussions page](gh:discussions).
- Raising [issues](gh:issues) related to bugs or desired enhancements.
- Contributing or improving the [docs](gh:tree/main/docs) or
  [examples](gh:tree/main/docs/examples).
- Fixing outstanding [issues](gh:issues) (bugs).
- Extending or improving our [codebase](https://github.com/QuantClimate/GPJax).


## Code of conduct

As a contributor to GPJax, you can help us keep the community open and
inclusive. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Opening issues and getting support

Please open issues on [Github Issue Tracker](gh:issues/new/choose). Here you can
mention

You can ask a question or start a discussion in the [Discussion
section](gh:discussions) on Github.

## Contributing to the source code

Submitting code contributions to GPJax is done via a [GitHub pull
request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
Our preferred workflow is to first fork the [GitHub repository](https://github.com/QuantClimate/GPJax), clone it to
your local machine, and develop on a _feature branch_. Once you're happy with your changes,
install our `pre-commit hooks`, `commit` and `push` your code.

**New to this?** Don't panic, our [guide](#step-by-step-guide) below will walk
you through every detail!

:::{note}
Before opening a pull request we recommend you check our [pull request checklist](#pull-request-checklist).
:::


### Step-by-step guide:

1.  Click [here](gh:fork) to Fork GPJax's
  codebase (alternatively, click the 'Fork' button towards the top right of
  the [main repository page](https://github.com/QuantClimate/GPJax)). This
  adds a copy of the codebase to your GitHub user account.

2.  Clone your GPJax fork from your GitHub account to your local disk, and add
  the base repository as a remote:
  ```bash
  $ git clone git@github.com:<your GitHub handle>/GPJax.git
  $ cd GPJax
  $ git remote add upstream git@github.com:QuantClimate/GPJax.git
  ```

3.  Create a `feature` branch to hold your development changes:

  ```bash
  $ git checkout -b my-feature
  ```
  Always use a `feature` branch. It's good practice to avoid
  work on the ``main`` branch of any repository.

4.  We use [uv](https://docs.astral.sh/uv/) for packaging and dependency management. Project requirements are in ``pyproject.toml``. To install GPJax with uv, run:

  ```bash
  $ uv sync
  ```

  Then install the pre-commit hooks. [prek](https://github.com/j178/prek) runs
  them before each commit. They run `ruff check`, `ruff format` and `ty`, the
  same checks as CI:

  ```bash
  $ uv run prek install
  ```

  At this point we recommend you check your installation passes the supplied unit tests:

  ```bash
  $ uv run poe all-tests
  ```

5.  Add changed files using `git add` and then `git commit` files to record your
  changes locally:

  ```bash
  $ git add modified_files
  $ git commit
  ```
  After committing, it is a good idea to sync with the base repository in case
  there have been any changes:

  ```bash
  $ git fetch upstream
  $ git rebase upstream/main
  ```

  Then push the changes to your GitHub account with:

  ```bash
  $ git push -u origin my-feature
  ```

6.  Go to the GitHub web page of your fork of the GPJax repo. Click the 'Pull
  request' button to send your changes to the project's maintainers for
  review.

### Changing the public API

Other packages use GPJax's public API. For example, skgp calls `gpx.fit`,
`gpx.gps.Prior` and `gpx.Dataset`. Obey this rule when you rename or remove a
public function, class, argument or module:

1. Add the new name. Keep the old name, and make it raise a
   `DeprecationWarning` that gives the new name.
2. Remove the old name only in a major release (for example, 2.0.0), and only
   after skgp stops using it.

skgp's tests fail on a GPJax `DeprecationWarning`. Thus skgp changes to the new
name before GPJax removes the old name.

### Pull request checklist

We welcome both complete or "work in progress" pull requests. Before opening
one, we recommended you check the following guidelines to ensure a smooth review
process.

**My contribution is a "work in progress":**

Please prefix the title of incomplete contributions with `[WIP]` (to indicate a
work in progress). WIPs are useful to:

  1. Indicate you are working on something to avoid duplicated work.
  2. Request broad review of functionality or API.
  3. Seek collaborators.

In the description of the pull request, we recommend you outline where work
needs doing. For example, do some tests need writing?

**My contribution is complete:**

If addressing an issue, please use the pull request title to describe the issue
and mention the issue number in the pull request description. This will make
sure a link back to the original issue is created. Then before making your pull
request, we recommend you check the following:

  - Do all public methods have informative docstrings that describe their
  function, input(s) and output(s)?
  - Do the pre-commit hooks pass?
  - Do the tests pass when everything is rebuilt from scratch?
  - Documentation and high-coverage tests are necessary for enhancements to be
  accepted. Test coverage can be checked with:

    ```bash
    $ uv run poe coverage
    ```

  Navigate to the newly created folder `htmlcov` and open `index.html` to view
  the coverage report.

This guide was derived from [PyMC's guide to
contributing](https://github.com/pymc-devs/pymc/blob/main/CONTRIBUTING.md).
