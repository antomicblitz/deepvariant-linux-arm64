# Packaging Guide: Homebrew Tap & Bioconda

This document explains how to distribute the macOS ARM64 build of DeepVariant via Homebrew and Bioconda.

## Overview

| Channel | Audience | Install command | Pros | Cons |
|---------|----------|----------------|------|------|
| **Homebrew tap** | macOS users | `brew install antomicblitz/deepvariant/deepvariant` | Full control, no approval needed, tensorflow-metal included | macOS only |
| **Bioconda** | Cross-platform bioinformatics | `conda install -c bioconda deepvariant` | Existing package (82K downloads), cross-platform | Needs maintainer approval, tensorflow-metal via pip is awkward |
| **install.sh** (current) | Everyone | `curl \| bash` | Works now, most flexible | No package manager integration |

## Homebrew Tap

### Setup (one-time)

```bash
# 1. Create the tap repository
brew tap-new antomicblitz/deepvariant

# 2. Copy the formula
cp packaging/homebrew/Formula/deepvariant.rb \
   "$(brew --repository antomicblitz/deepvariant)/Formula/"

# 3. Push to GitHub
cd "$(brew --repository antomicblitz/deepvariant)"
git add -A && git commit -m "Add deepvariant formula v1.9.0"
gh repo create antomicblitz/homebrew-deepvariant --push --public --source .
```

### How it works

The formula (`packaging/homebrew/Formula/deepvariant.rb`):

1. Downloads the pre-built tarball from GitHub Releases
2. Creates a Python 3.10 virtualenv in `$(brew --prefix)/Cellar/deepvariant/<version>/libexec/venv/`
3. Installs `tensorflow-macos` + `tensorflow-metal` + all DeepVariant Python deps via pip
4. Installs pre-built Bazel binaries (C++ extensions + Python zip packages)
5. Creates wrapper scripts in `bin/` that invoke the venv Python
6. Downloads the WGS model on first install via `post_install`

### User experience

```bash
# Install
brew tap antomicblitz/deepvariant
brew install deepvariant

# Use
run_deepvariant \
  --model_type WGS \
  --ref reference.fasta \
  --reads input.bam \
  --output_vcf output.vcf \
  --num_shards $(sysctl -n hw.perflevel0.logicalcpu)

# Uninstall
brew uninstall deepvariant
brew untap antomicblitz/deepvariant
```

### Testing locally

```bash
# Install from local formula (no tap required)
brew install --formula packaging/homebrew/Formula/deepvariant.rb

# Run tests
brew test deepvariant

# Build a bottle for distribution
brew install --build-bottle antomicblitz/deepvariant/deepvariant
brew bottle antomicblitz/deepvariant/deepvariant
```

### Bottles (pre-built)

Homebrew "bottles" are pre-built packages that skip the install-time pip installs. To create and host them:

1. Run `brew install --build-bottle` on a clean machine
2. Run `brew bottle` to produce a `.tar.gz`
3. Upload the bottle to GitHub Releases
4. Add the `bottle do ... end` block to the formula

This makes `brew install` nearly instant (no pip install at all). Without a bottle, the first install takes ~5 minutes for pip to resolve and install everything.

### Updating for new versions

1. Build new binaries: `./build_release_binaries.sh`
2. Package: `./scripts/package_release.sh --release`
3. Update the formula:
   - Change `version`
   - Update the `url` to point to the new release
   - Update the `sha256` (`shasum -a 256 deepvariant-X.Y.Z-macos-arm64.tar.gz`)

---

## Bioconda

### Background

DeepVariant already has a bioconda recipe (`recipes/deepvariant/`) but it's essentially non-functional:
- Declared `noarch: python` but all binary-handling code is commented out
- Wrapper scripts contain placeholder paths that are never substituted
- Tests are completely disabled
- Dependencies are outdated (pins tensorflow 2.11)

### Strategy

Rather than building from source in bioconda CI (impractical due to Bazel 5.3.0 + 30 GB TF source), we use the same pre-built binary approach but make it platform-aware:

- **linux-64**: Keep existing wrapper-based approach (to be improved separately)
- **osx-arm64**: Download pre-built tarball from our GitHub Release

### How to submit

```bash
# 1. Fork bioconda-recipes
gh repo fork bioconda/bioconda-recipes --clone
cd bioconda-recipes

# 2. Create a branch
git checkout -b deepvariant-osx-arm64

# 3. Replace the recipe files
cp /path/to/packaging/bioconda/meta.yaml recipes/deepvariant/meta.yaml
cp /path/to/packaging/bioconda/build.sh recipes/deepvariant/build.sh

# 4. Test locally (requires bioconda-utils)
pip install bioconda-utils
bioconda-utils lint --packages deepvariant
bioconda-utils build --packages deepvariant

# 5. Open PR
gh pr create --title "deepvariant: add osx-arm64 support via pre-built binaries" \
  --body "Adds native macOS ARM64 (Apple Silicon) support to DeepVariant.

## Changes
- Remove \`noarch: python\` — now platform-specific
- Add \`osx-arm64\` to additional-platforms
- osx-arm64 build downloads pre-built native binaries from GitHub Release
- Add tensorflow-macos + tensorflow-metal for Metal GPU acceleration (~4.25x speedup)
- Add real tests (verify binaries and TF import)
- Update Python pin to 3.10 (matches upstream requirement)
- Update dependencies

## Testing
- Verified on M1 Max: full pipeline runs successfully
- Metal GPU provides 4.25x speedup for call_variants inference
- Accuracy matches published benchmarks (SNP F1 ~0.9996, INDEL F1 ~0.9966)"
```

### Key differences from the Homebrew formula

| Aspect | Homebrew | Bioconda |
|--------|----------|----------|
| tensorflow-metal | Installed via pip in venv | Installed via pip in build.sh |
| Python env | Dedicated venv | Conda environment |
| Model download | post_install hook | User runs manually |
| Approval | None (your own tap) | Bioconda maintainer review |

### tensorflow-metal in conda

`tensorflow-metal` is only available via pip (Apple publishes no conda package). The bioconda build.sh installs it with `pip install --no-deps` inside the conda environment. This is a known pattern in bioconda for pip-only packages.

If a bioconda reviewer objects to pip-installing tensorflow-metal, the fallback is to skip it and document it as an optional post-install step. However, this would cost users a 4.25x performance regression, so it's worth advocating for the pip install approach.

### Known limitations

1. **Linux build is still broken.** The existing Linux recipe installs non-functional wrapper scripts. Fixing Linux is a separate effort (probably requires Google to publish pre-built linux-64 binaries for v1.9.0).

2. **tensorflow-metal is closed-source.** It's an Apple binary wheel. Cannot be built from source or re-packaged as a conda package.

3. **Bioconda CI time limits.** The pip installs take ~3-5 minutes. Bioconda CI has generous limits (~30 min) so this should be fine, but if it times out, pre-downloading wheels into the recipe would help.

---

## Recommendation

**Start with the Homebrew tap.** It's faster to set up, gives you full control, and provides the best user experience (tensorflow-metal included, model auto-download, `brew upgrade` support). You can have it working today.

**Submit the bioconda PR in parallel.** It takes longer (bioconda review process), and the tensorflow-metal question may require discussion with maintainers. But having DeepVariant in bioconda with real osx-arm64 support would benefit the broader bioinformatics community.
