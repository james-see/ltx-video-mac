// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LTXVideoGenerator",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/pvieito/PythonKit.git", revision: "1ac25ddddca845ba0184911de9c00e1508948cf8")
    ],
    targets: [
        .executableTarget(
            name: "LTXVideoGenerator",
            dependencies: ["PythonKit"],
            path: "Sources"
        )
    ]
)
