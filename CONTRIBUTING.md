## Contributing to OpenSCvx

### Local Installation

To set up OpenSCvx for local development, first clone the repository and create a virtual environment:

```bash
git clone https://github.com/haynec/OpenSCvx.git
cd OpenSCvx
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install the package in editable mode with development dependencies:

```bash
pip install -e ".[test]"
```

For additional features, you can install optional dependencies:
- `pip install -e ".[gui]"` - GUI visualization tools
- `pip install -e ".[cvxpygen]"` - Code generation support
- `pip install -e ".[stl]"` - STL (Signal Temporal Logic) constraints

#### Using uv (Alternative)

If you prefer using [uv](https://github.com/astral-sh/uv) for faster dependency resolution:

```bash
git clone https://github.com/haynec/OpenSCvx.git
cd OpenSCvx
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[test]"
```

To contribute changes, you'll need to create a branch or fork. See [Forking and Merging](#forking-and-merging) below for details on our workflow and [Branch Naming Conventions](#branch-naming-conventions) for required naming patterns.

### Forking and Merging

All contributions must be submitted via pull request and approved by a maintainer before merging. This ensures code quality, maintains consistency across the codebase, and gives contributors feedback on their changes.

We currently use standard merge commits for pull requests. We may transition to squash merges in the future to maintain a cleaner commit history on `main`.

Contributors can either:
1. **Create a branch directly** on the OpenSCvx repository (for those with write access)
2. **Fork the repository** to their own GitHub account and submit pull requests from there (recommended for external contributors)

#### Fork Setup

To set up a fork for development:

1. Fork the repository on GitHub by clicking the "Fork" button on the [OpenSCvx repository page](https://github.com/haynec/OpenSCvx)

2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OpenSCvx.git
   cd OpenSCvx
   ```

3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/haynec/OpenSCvx.git
   ```

4. Verify your remotes:
   ```bash
   git remote -v
   # origin    https://github.com/YOUR_USERNAME/OpenSCvx.git (fetch)
   # origin    https://github.com/YOUR_USERNAME/OpenSCvx.git (push)
   # upstream  https://github.com/haynec/OpenSCvx.git (fetch)
   # upstream  https://github.com/haynec/OpenSCvx.git (push)
   ```

5. Keep your fork up to date by periodically pulling from upstream:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

#### Branch Naming Conventions

We use [release-drafter](https://github.com/release-drafter/release-drafter) to automatically generate release notes based on pull request labels. These labels are auto-assigned based on your branch name prefix.

**Required branch prefixes:**

| Prefix | Label Applied | Release Category |
|--------|---------------|------------------|
| `feature/` | enhancement | Features |
| `fix/` | bug | Bug Fixes |
| `refactor/` | refactor | Maintenance |
| `chore/` | chore | Maintenance |
| `docs/` or `doc/` | documentation | Maintenance |

**Examples:**
- `feature/add-new-solver` - Adding a new solver option
- `fix/convergence-issue` - Fixing a bug
- `refactor/simplify-dynamics` - Code refactoring
- `chore/update-dependencies` - Maintenance tasks
- `docs/api-examples` - Documentation updates

See the [workflow](.github/workflows/release-drafter.yml) and [configuration](.github/release-drafter.yml) for details.

### Development Guidelines

#### Use of LLM-Based Tools

While the development of OpenSCvx has been greatly assisted by LLM-based tools for coding tasks, it is up to the contributor to use them in a responsible manner. We welcome their use, _but with great power comes great responsibility_.

Our goal is to produce a high-quality codebase that is well organized, easy to modify, easy to understand, and, most importantly, easy to maintain. We will not accept so-called _AI-slop_. Contributors are expected to:

1. **Understand every line** of code they submit. You are responsible for it, not the LLM
2. **Review and refactor** generated code to match the project's style and patterns
3. **Write meaningful commit messages and keep commits small-ish** to break the changes into bite-sized chunks that can be understood based on the commit message.
4. **Test thoroughly** - LLMs can generate plausible-looking but subtly incorrect code. See point 1.

#### Keep It Simple

We value simplicity over cleverness. Before adding code, ask yourself:

- **[YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)**: "You Aren't Gonna Need It" - don't add functionality until it's actually needed
- **[KISS](https://en.wikipedia.org/wiki/KISS_principle)**: "Keep It Simple, Stupid" - the simplest solution is usually the best one
- **[Feature creep](https://en.wikipedia.org/wiki/Feature_creep)**: the enemy of maintainable software. A 10-line function that's easy to understand beats a 50-line "flexible" abstraction that handles hypothetical future requirements. When in doubt, leave it out.

While we don't always follow these rules, we try.

#### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before submitting a PR, ensure your code passes:

```bash
ruff check .
ruff format --check .
```

To auto-fix issues:

```bash
ruff check --fix .
ruff format .
```

See the `[tool.ruff]` section in `pyproject.toml` for our configuration.

#### Testing

OpenSCvx uses the `pytest` framework to organize the automated testing.

##### Brachistochrone Problem

The [brachistochrone problem](https://en.wikipedia.org/wiki/Brachistochrone_curve) serves as our primary validation benchmark. This classic problem asks for the curve of fastest descent under gravity between two points, and has a known analytical solution: the cycloid. We use it as part of the unit-test suite because:

- **Analytical solution exists**: We can validate numerical results against the exact cycloid curve
- **Small and fast**: The problem converges quickly, making it ideal for CI/CD pipelines
- **Representative**: It exercises core features like free final time optimization and nonlinear dynamics

Run the test suite with:

```bash
pytest tests/
```

For faster iteration during development, run only the brachistochrone tests:

```bash
pytest tests/test_brachistochrone.py -v
```

##### Test-Driven Development

While we do not strictly follow [test driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development), we aim to keep a representative test-suite where it makes sense to do so and leverage TDD when possible.

For example, the [symbolic-expression layer](openscvx/symbolic/__init__.py) is well covered by [tests](tests/symbolic/) which were written simultaneously with the symbolic layer.

The extensive test suite helps ensure that new features and refactors don't break existing functionality.
