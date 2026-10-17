//! Versioned exit negotiation. All fields are inside the Noise handshake.
use std::net::{Ipv4Addr, Ipv6Addr};

use crate::error::{Error, Result};
use crate::types::SessionId;

/// A reconnect advertises the existing lease and next return sequence, so an
/// exit restart cannot silently change addresses or reuse delivered sequences.
#[derive(Debug, Clone)]
pub(crate) struct ExitHello {
    pub token: [u8; 32],
    pub session: SessionId,
    pub uplink: u16,
    pub preferred_ipv4: Option<Ipv4Addr>,
    pub next_return_sequence: u64,
    pub path_epoch: u64,
    pub access_token: Option<[u8; 32]>,
    pub version: u8,
}

impl ExitHello {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = b"TG3".to_vec();
        bytes.extend_from_slice(&self.token);
        bytes.extend_from_slice(&self.session.0);
        bytes.extend_from_slice(&self.uplink.to_be_bytes());
        bytes.extend_from_slice(
            &self
                .preferred_ipv4
                .unwrap_or(Ipv4Addr::UNSPECIFIED)
                .octets(),
        );
        bytes.extend_from_slice(&self.next_return_sequence.to_be_bytes());
        bytes.extend_from_slice(&self.path_epoch.to_be_bytes());
        if let Some(token) = self.access_token {
            bytes.extend_from_slice(&token);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (version, size) = match bytes.get(..3) {
            Some(b"TG2") => (2, 69),
            Some(b"TG3") => (3, 89),
            _ => return Err(Error::Authentication("unsupported exit handshake".into())),
        };
        if bytes.len() != size && bytes.len() != size + 32 {
            return Err(Error::Authentication(
                "invalid exit handshake length".into(),
            ));
        }
        let preferred = if version == 3 {
            Ipv4Addr::from(<[u8; 4]>::try_from(&bytes[69..73]).unwrap())
        } else {
            Ipv4Addr::UNSPECIFIED
        };
        Ok(Self {
            token: bytes[3..35].try_into().unwrap(),
            session: SessionId(bytes[35..67].try_into().unwrap()),
            uplink: u16::from_be_bytes(bytes[67..69].try_into().unwrap()),
            preferred_ipv4: (!preferred.is_unspecified()).then_some(preferred),
            next_return_sequence: if version == 3 {
                u64::from_be_bytes(bytes[73..81].try_into().unwrap())
            } else {
                0
            },
            path_epoch: if version == 3 {
                u64::from_be_bytes(bytes[81..89].try_into().unwrap())
            } else {
                0
            },
            access_token: (bytes.len() == size + 32).then(|| bytes[size..].try_into().unwrap()),
            version,
        })
    }
}

pub(crate) fn encode_exit_reply(version: u8, ipv4: Ipv4Addr, ipv6: Option<Ipv6Addr>) -> Vec<u8> {
    let mut bytes = if version == 3 { b"TG3" } else { b"TG2" }.to_vec();
    bytes.extend_from_slice(&ipv4.octets());
    if let Some(ipv6) = ipv6 {
        bytes.extend_from_slice(&ipv6.octets());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_negotiation_preserves_lease_and_sequence() {
        for access_token in [None, Some([0x57; 32])] {
            let hello = ExitHello {
                token: [0x31; 32],
                session: SessionId::generate(),
                uplink: 17,
                preferred_ipv4: Some(Ipv4Addr::new(10, 0, 85, 7)),
                next_return_sequence: 0xffff0000abcd1234,
                path_epoch: 42,
                access_token,
                version: 3,
            };
            let encoded = hello.encode();
            let decoded = ExitHello::decode(&encoded).unwrap();
            assert_eq!(decoded.session, hello.session);
            assert_eq!(decoded.uplink, hello.uplink);
            assert_eq!(decoded.preferred_ipv4, hello.preferred_ipv4);
            assert_eq!(decoded.next_return_sequence, hello.next_return_sequence);
            assert_eq!(decoded.path_epoch, hello.path_epoch);
            assert_eq!(decoded.access_token, access_token);
            for end in 0..encoded.len() {
                // The only valid truncated form omits the optional access token.
                if access_token.is_none() || end != 89 {
                    assert!(ExitHello::decode(&encoded[..end]).is_err(), "{end}");
                }
            }
            let mut extra = encoded;
            extra.push(0);
            assert!(ExitHello::decode(&extra).is_err());
        }
    }

    #[test]
    fn old_clients_remain_decodable_without_claiming_restart_continuity() {
        let mut bytes = b"TG2".to_vec();
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&[2; 32]);
        bytes.extend_from_slice(&3u16.to_be_bytes());
        let hello = ExitHello::decode(&bytes).unwrap();
        assert_eq!(hello.version, 2);
        assert_eq!(hello.preferred_ipv4, None);
        assert_eq!(hello.next_return_sequence, 0);
    }
}
