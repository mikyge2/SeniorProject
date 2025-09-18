#!/usr/bin/env python3
"""
Simple test script to verify espeak is working correctly.
"""

import subprocess
import shutil
import sys
import time


def test_espeak_installation():
    """Test if espeak is properly installed and working."""
    print("=" * 50)
    print("ESPEAK INSTALLATION TEST")
    print("=" * 50)

    # Check if espeak command exists
    espeak_path = shutil.which('espeak')
    if not espeak_path:
        print("❌ ERROR: espeak command not found in PATH")
        print("   Install espeak with:")
        print("   • Arch/Manjaro: sudo pacman -S espeak espeak-data")
        print("   • Fedora: sudo dnf install espeak espeak-devel")
        print("   • Ubuntu/Debian: sudo apt-get install espeak espeak-data")
        return False

    print(f"✅ espeak found at: {espeak_path}")

    # Test espeak version
    try:
        result = subprocess.run(['espeak', '--version'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ espeak version: {result.stdout.strip()}")
        else:
            print(f"⚠️  espeak --version returned code {result.returncode}")
    except Exception as e:
        print(f"⚠️  Could not get espeak version: {e}")

    return True


def test_espeak_speech():
    """Test actual speech output."""
    print("\n" + "=" * 50)
    print("ESPEAK SPEECH TEST")
    print("=" * 50)

    test_phrases = [
        "Hello world",
        "Testing espeak",
        "ASL detection system ready"
    ]

    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n🔊 Test {i}: Speaking '{phrase}'...")
        print("   (You should hear this through your speakers)")

        try:
            # Use the same command structure as the ASL system
            cmd = ['espeak', '-s150', '-a80', '-g5', phrase]
            result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ espeak command successful")
            else:
                print(f"❌ espeak failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ espeak timed out speaking '{phrase}'")
            return False
        except Exception as e:
            print(f"❌ Error running espeak: {e}")
            return False

        # Pause between tests
        time.sleep(0.5)

    return True


def test_audio_system():
    """Test if audio system is working."""
    print("\n" + "=" * 50)
    print("AUDIO SYSTEM CHECK")
    print("=" * 50)

    try:
        # Check for PulseAudio
        result = subprocess.run(['pactl', 'info'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ PulseAudio is running")
        else:
            print("⚠️  PulseAudio not detected")
    except:
        print("⚠️  Could not check PulseAudio status")

    try:
        # List audio devices
        result = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            print("✅ Audio output devices found:")
            for line in result.stdout.split('\n')[:3]:  # Show first 3 devices
                if line.strip():
                    print(f"   {line}")
        else:
            print("⚠️  No audio output devices found")
    except:
        print("⚠️  Could not list audio devices")


def main():
    """Run all tests."""
    print("ESPEAK & AUDIO DIAGNOSTIC TOOL")
    print("This will test if espeak can produce audio output")
    print()

    # Test espeak installation
    if not test_espeak_installation():
        print("\n❌ ESPEAK INSTALLATION FAILED")
        print("Please install espeak and try again.")
        return 1

    # Test audio system
    test_audio_system()

    # Test actual speech
    if not test_espeak_speech():
        print("\n❌ ESPEAK SPEECH TEST FAILED")
        print("Check your audio system and try:")
        print("1. Adjust system volume")
        print("2. Check audio device settings")
        print("3. Try: espeak 'test' (manually)")
        return 1

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("espeak is working correctly and should work in the ASL system.")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())