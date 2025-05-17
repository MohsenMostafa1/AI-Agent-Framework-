# Model Context Protocol (MCP) Integration Guide

## Core Components
1. **Protocol Version**: v2.3.1 (GitHub-hosted reference implementation)
2. **Transport Layer**: 
   - HTTPS for control plane
   - WebSockets (RFC 6455) for data plane
3. **Message Format**:
   ```protobuf
   message MCPFrame {
     string context_id = 1;
     bytes payload = 2;
     map<string, string> metadata = 3;
   }
