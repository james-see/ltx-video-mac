import Darwin
import Foundation

/// Host memory stats for lightweight preflight checks (not a guarantee of peak MLX usage).
enum MacOSSystemMemory {
    static var physicalMemoryBytes: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Approximate currently reusable RAM: free + inactive file-backed pages (vm_statistics64).
    static func approximateAvailableBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return physicalMemoryBytes / 4
        }
        let pageSize = UInt64(vm_kernel_page_size)
        let free = UInt64(stats.free_count) * pageSize
        let inactive = UInt64(stats.inactive_count) * pageSize
        return free + inactive
    }

    static func physicalMemoryGBFormatted() -> String {
        String(format: "%.0f", Double(physicalMemoryBytes) / 1_073_741_824.0)
    }

    static func approximateAvailableMemoryGBFormatted() -> String {
        String(format: "%.1f", Double(approximateAvailableBytes()) / 1_073_741_824.0)
    }
}
