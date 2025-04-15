# Comrade Installation and Usage Guide

## Overview

Before diving in, make sure you’re familiar with the installation quirks that might trip you up. This guide covers:

- **Installation via Juliaup:** The official and hassle-free way to install Julia and its libraries.
- **Package & Version Management:** Keeping your versions just right—especially important since Comrade uses the Enzyme package for high-speed calculations.
- **Project Environment Setup:** How to prevent your version info from getting dumped into the base environment.

---

## 1. Installing Julia with Juliaup

The Comrade team recommends using [Juliaup](https://github.com/JuliaLang/Juliaup) to manage your Julia installations and libraries automatically. Juliaup helps you keep everything in sync, including the versioning for both Julia and its packages.

- **Heads-up:** The command you need to run in your terminal varies by operating system. Check out the instructions in the Juliaup GitHub README for the correct command on your OS.

---

## 2. Managing Packages and Versions

Because Comrade relies on the Enzyme package for real-time automatic differentiation (which really speeds up your computations), it’s critical to use the right Julia version. Enzyme does some low-level stuff, so if you’re on a version of Julia that's too fresh, you might run into issues.

- **What to do:** Always refer to the official documentation for the recommended Julia version. As a reference, here are the versions I’m using (as of January 2025):

```julia
Julia Version 1.10.6
[13f3f980] CairoMakie v0.12.16
[99d987ce] Comrade v0.11.2
[0b91fe84] DisplayAs v0.1.6
[31c24e10] Distributions v0.25.113
[682c06a0] JSON v0.21.4
[7f7a1694] Optimization v4.0.5
[3e6eede4] OptimizationBBO v0.4.0
[0eb8d820] Pigeons v0.4.7
[91a5bcdd] Plots v1.40.9
[3d61700d] Pyehtim v0.1.3
[860ef19b] StableRNGs v1.0.2
[09ab397b] StructArrays v0.6.21
[b1ba175b] VLBIImagePriors v0.9.1
[d6343c73] VLBISkyModels v0.6.5
```

---

## 3. Installing Additional Packages

With Juliaup, adding packages is a breeze. Just open your Julia REPL and run:

```julia
using Pkg
Pkg.add(PackageSpec(name="package_name", version="version_number"))
```

- **Important Note:** If you try to load a package that hasn’t been installed, you’ll get an error. So, be sure to add it first.

---

## 4. Setting Up Your Project Environment

To keep things tidy, Juliaup saves package version information to a `toml` file within your project directory. This prevents your main (base) environment from getting cluttered with project-specific versions.

- **How to activate your project environment:**  
  In your working directory, start Julia and activate the environment:

  ```julia
  Pkg.activate(".")
  ```

  This ensures all version info is stored locally in your project’s `toml` file.

---
