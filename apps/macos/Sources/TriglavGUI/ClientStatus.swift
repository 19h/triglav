import Foundation

enum ClientStatusEndpoint {
    static let address = "127.0.0.1:9091"
    static let statusURL = URL(string: "http://\(address)/status")!
}

struct ClientStatusSnapshot: Decodable, Equatable {
    let version: String
    let uptimeSeconds: UInt64
    let state: String
    let role: String?
    let mode: String?
    let processId: UInt32?
    let sessionId: String?
    let connectionId: String?
    let quality: ClientQualitySnapshot?
    let tunnel: ClientTunnelSnapshot?
    let uplinks: [ClientUplinkSnapshot]
    let sessions: [ClientSessionSnapshot]
    let totalBytesSent: UInt64
    let totalBytesReceived: UInt64
    let totalConnections: UInt64
}

struct ClientQualitySnapshot: Decodable, Equatable {
    let usableUplinks: Int
    let totalUplinks: Int
    let avgRttMs: Double
    let avgLossPercent: Double
    let totalBandwidthMbps: Double
    let packetsSent: UInt64
    let packetsReceived: UInt64
    let packetsDropped: UInt64
}

struct ClientTunnelSnapshot: Decodable, Equatable {
    let tunName: String
    let fullTunnel: Bool
    let includeRoutes: [String]
    let excludeRoutes: [String]
}

struct ClientUplinkSnapshot: Decodable, Identifiable, Equatable {
    let id: String
    let interface: String?
    let remoteAddr: String
    let state: String
    let health: String
    let rttMs: Double
    let lossPercent: Double
    let bandwidthMbps: Double
    let bytesSent: UInt64
    let bytesReceived: UInt64
    let natType: String
    let externalAddr: String?
}

struct ClientSessionSnapshot: Decodable, Identifiable, Equatable {
    let id: String
    let userId: String?
    let remoteAddrs: [String]
    let connectedAt: String
    let bytesSent: UInt64
    let bytesReceived: UInt64
    let uplinksUsed: [String]
}

extension ClientStatusSnapshot {
    var isClientRole: Bool {
        (role ?? "client") == "client"
    }
}

extension JSONDecoder {
    static let triglavStatusDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()
}

extension UInt64 {
    var byteCountDisplay: String {
        ByteCountFormatter.string(fromByteCount: Int64(self), countStyle: .binary)
    }
}
