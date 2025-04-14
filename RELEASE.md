# Release Process

This document describes the process for releasing new versions of the `flowtracks` package.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backward compatible manner
- **PATCH** version when you make backward compatible bug fixes

## Bumping the Version

To bump the version, use the provided script:

```bash
# Bump the patch version (e.g., 1.0.1 -> 1.0.2)
python bump_version.py patch

# Bump the minor version (e.g., 1.0.1 -> 1.1.0)
python bump_version.py minor

# Bump the major version (e.g., 1.0.1 -> 2.0.0)
python bump_version.py major
```

## Release Process

1. Make sure all your changes are committed and pushed to the repository.

2. Bump the version using the script above.

3. Commit the version change:
   ```bash
   git add flowtracks/__init__.py
   git commit -m "Bump version to x.y.z"
   ```

4. Create a tag for the new version:
   ```bash
   git tag -a vx.y.z -m "Release version x.y.z"
   ```

5. Push the changes and the tag:
   ```bash
   git push origin master
   git push origin vx.y.z
   ```

6. GitHub Actions will automatically:
   - Run tests on the tagged version
   - Build the package
   - Upload it to PyPI (if tests pass)

## PyPI Configuration

To enable automatic publishing to PyPI, you need to:

1. Create an API token on PyPI:
   - Go to https://pypi.org/manage/account/
   - Create an API token with scope "Upload to project"

2. Add the token as a secret in your GitHub repository:
   - Go to your repository on GitHub
   - Navigate to Settings > Secrets > Actions
   - Create a new secret named `PYPI_API_TOKEN` with the value of your PyPI token
