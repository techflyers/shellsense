#!/usr/bin/env python3
"""
Universal Package Manager Detector
Checks for common system and language-specific package managers across Linux, macOS, Windows, and BSD variants.
Provides a summary function for programmatic use.
"""
import shutil
import platform

# Mapping of package manager commands to descriptions
dicts = {
    # System-level (Linux)
    "apt": "Debian/Ubuntu based",
    "apt-get": "Debian/Ubuntu based (lower-level)",
    "dpkg": "Debian/Ubuntu based (core package tool)",
    "yum": "Older RHEL/Fedora/CentOS",
    "dnf": "Modern RHEL/Fedora/CentOS",
    "rpm": "RHEL/Fedora/CentOS (core package tool)",
    "pacman": "Arch Linux based",
    "zypper": "SUSE/openSUSE",
    "emerge": "Gentoo",
    "apk": "Alpine Linux",
    # Cross-distro & macOS
    "snap": "Snapcraft (Canonical)",
    "flatpak": "Flatpak",
    "brew": "Homebrew (macOS/Linux)",
    "nix-env": "Nix package manager",
    # Windows
    "choco": "Chocolatey (Windows)",
    "winget": "WinGet (Windows)",
    "scoop": "Scoop (Windows)",
    # BSD
    "pkg": "Termux apt wrapper / FreeBSD pkg",
    "pkg_add": "OpenBSD pkg_add",
    "pkgin": "NetBSD pkgin",
    # Language-specific / Environment
    "pip": "Python",
    "pip3": "Python 3",
    "conda": "Anaconda/Miniconda (Python, R, etc.)",
    "npm": "Node.js (JavaScript)",
    "yarn": "Node.js (alternative to npm)",
    "gem": "Ruby",
    "composer": "PHP",
    "cargo": "Rust",
    "go": "Go (go get/install)",
    "cpan": "Perl (CPAN)",
}

# Translation map to clean up descriptions for summary
_summary_trans = str.maketrans({
    "(": "-",
    ")": "",
    "/": "",
    " ": "",
    "\n": "",
})

def detect_pkg_managers():
    """
    Detects available package managers and returns a list of tuples (cmd, desc).
    """
    found = []
    for cmd, desc in dicts.items():
        if shutil.which(cmd):
            found.append((cmd, desc))
    return found


def summarize_pkg_managers():
    """
    Returns a comma-separated string of detected managers in the form:
    cmd-sanitizedDesc, ...
    where sanitizedDesc has no spaces or special chars.
    """
    entries = []
    for cmd, desc in detect_pkg_managers():
        clean_desc = desc.translate(_summary_trans)
        entries.append(f"{cmd}-{clean_desc}")
    return ", ".join(entries), platform.system(), platform.machine()


def main():
    found_any = False
    print("Checking for installed package managers...")
    print("----------------------------------------")

    for cmd, desc in detect_pkg_managers():
        print(f"[+] Found: {cmd} ({desc})")
        found_any = True

    print("----------------------------------------")
    if not found_any:
        print("No common package managers from the checked list were found in your PATH.")
    else:
        print("Note: This list reflects commands available in your PATH.")

    # Display summary string
    print(f"Summary: {summarize_pkg_managers()[0]}")

    # Suggest primary system type based on detections
    os_name = platform.system()
    print(f"Detected OS: {os_name}")

    if os_name == "Linux":
        if shutil.which("apt") or shutil.which("dpkg"):
            print("  Likely Debian/Ubuntu based (apt/dpkg)")
        elif shutil.which("dnf") or shutil.which("rpm"):
            print("  Likely Fedora/RHEL based (dnf/rpm)")
        elif shutil.which("yum"):
            print("  Likely older RHEL/CentOS based (yum/rpm)")
        elif shutil.which("pacman"):
            print("  Likely Arch Linux based (pacman)")
        elif shutil.which("zypper"):
            print("  Likely SUSE/openSUSE based (zypper)")
        elif shutil.which("apk"):
            print("  Likely Alpine Linux based (apk)")
        else:
            print("  Could not reliably determine Linux distro from common indicators.")
    elif os_name == "Darwin":
        print("  macOS detected (use Homebrew or MacPorts).")
    elif os_name == "Windows":
        if shutil.which("choco"):
            print("  Chocolatey detected (choco)")
        elif shutil.which("winget"):
            print("  WinGet detected (winget)")
        elif shutil.which("scoop"):
            print("  Scoop detected (scoop)")
        else:
            print("  Could not detect a Windows package manager (choco, winget, scoop).")
    elif os_name in ("FreeBSD", "OpenBSD", "NetBSD"):
        if os_name == "FreeBSD":
            print("  FreeBSD detected (pkg - FreeBSD package manager)")
        elif os_name == "OpenBSD":
            print("  OpenBSD detected (pkg_add - OpenBSD package manager)")
        elif os_name == "NetBSD":
            print("  NetBSD detected (pkgin - NetBSD package manager)")
    else:
        print("  Unrecognized or unsupported OS; detection may be incomplete.")


if __name__ == "__main__":
    main()
