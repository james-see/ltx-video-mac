#!/bin/bash
set -e

# LTX Video Generator - CI Release Build Script
# This script is designed to run in GitHub Actions

# Configuration
APP_NAME="LTX Video Generator"
BUNDLE_ID="com.jamescampbell.ltxvideogenerator"
SCHEME="LTXVideoGenerator"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
DIST_DIR="${PROJECT_DIR}/dist"
ARCHIVE_PATH="${BUILD_DIR}/${SCHEME}.xcarchive"
APP_PATH="${BUILD_DIR}/${APP_NAME}.app"

# Get version from tag or default
VERSION="${GITHUB_REF_NAME:-v1.0.0}"
VERSION="${VERSION#v}"  # Remove 'v' prefix
DMG_NAME="${SCHEME}-${VERSION}.dmg"
DMG_PATH="${DIST_DIR}/${DMG_NAME}"

echo "=== LTX Video Generator CI Build ==="
echo "Version: ${VERSION}"
echo "Building for: macOS"
echo ""

# Clean and setup directories
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

# Build the archive
echo "Building archive..."
cd "${PROJECT_DIR}/LTXVideoGenerator"
xcodebuild -project LTXVideoGenerator.xcodeproj \
    -scheme "${SCHEME}" \
    -configuration Release \
    -archivePath "${ARCHIVE_PATH}" \
    archive \
    CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY}" \
    DEVELOPMENT_TEAM="${APPLE_TEAM_ID}" \
    MARKETING_VERSION="${VERSION}"

# Export the app
echo "Exporting app..."
ditto "${ARCHIVE_PATH}/Products/Applications/${APP_NAME}.app" "${APP_PATH}"

# Verify signature
echo "Verifying signature..."
codesign -dv --verbose=2 "${APP_PATH}"

# Create zip for notarization
echo "Creating notarization package..."
ditto -c -k --keepParent "${APP_PATH}" "${BUILD_DIR}/app.zip"

# Submit for notarization
echo "Submitting for notarization..."
xcrun notarytool submit "${BUILD_DIR}/app.zip" \
    --apple-id "${APPLE_ID}" \
    --password "${APPLE_ID_PASSWORD}" \
    --team-id "${APPLE_TEAM_ID}" \
    --wait

# Staple
echo "Stapling ticket..."
xcrun stapler staple "${APP_PATH}"

# Create DMG
echo "Creating DMG..."
hdiutil create -volname "${APP_NAME}" \
    -srcfolder "${APP_PATH}" \
    -ov \
    -format UDZO \
    "${DMG_PATH}"

# Sign DMG
echo "Signing DMG..."
codesign --force --sign "${CODE_SIGN_IDENTITY}" "${DMG_PATH}"

# Notarize DMG
echo "Notarizing DMG..."
xcrun notarytool submit "${DMG_PATH}" \
    --apple-id "${APPLE_ID}" \
    --password "${APPLE_ID_PASSWORD}" \
    --team-id "${APPLE_TEAM_ID}" \
    --wait

# Staple DMG
xcrun stapler staple "${DMG_PATH}"

# Generate checksum
CHECKSUM=$(shasum -a 256 "${DMG_PATH}" | awk '{print $1}')
echo "${CHECKSUM}  ${DMG_NAME}" > "${DIST_DIR}/${DMG_NAME}.sha256"

echo ""
echo "=== Build Complete ==="
echo "DMG: ${DMG_PATH}"
echo "SHA256: ${CHECKSUM}"
