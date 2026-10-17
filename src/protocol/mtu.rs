//! Authenticated MTU control messages carried by production liveness probes.
use crate::error::{Error, Result};

pub(crate) const MIN_INNER_MTU: u16 = 1280;
pub(crate) const DEFAULT_INNER_MTU: u16 = 1340;
pub(crate) const MAX_INNER_MTU: u16 = (crate::protocol::MAX_PAYLOAD_SIZE - 33) as u16;

pub(crate) fn validate_mtu(mtu: u16) -> Result<u16> {
    if !(MIN_INNER_MTU..=MAX_INNER_MTU).contains(&mtu) {
        return Err(Error::Config(format!(
            "Inner MTU must be between {MIN_INNER_MTU} and {MAX_INNER_MTU} bytes"
        )));
    }
    Ok(mtu)
}

/// Bound an inner packet by Triglav, authenticated-record, UDP and IP headers.
pub(crate) fn path_inner_mtu(outer: u32, ipv6: bool) -> Result<u16> {
    let overhead = crate::protocol::HEADER_SIZE as u32 + 33 + 8 + if ipv6 { 40 } else { 20 };
    let inner = outer.saturating_sub(overhead).min(u32::from(MAX_INNER_MTU));
    if inner < u32::from(MIN_INNER_MTU) {
        return Err(Error::Config(format!("Outer MTU {outer} cannot carry a {MIN_INNER_MTU}-byte inner IPv6 packet without fragmentation")));
    }
    Ok(inner as u16)
}

pub(crate) fn request(challenge: &[u8; 16], mtu: u16) -> Vec<u8> {
    let mut payload = challenge.to_vec();
    payload.extend_from_slice(b"MTU1");
    payload.extend_from_slice(&mtu.to_be_bytes());
    payload
}

pub(crate) fn requested(payload: &[u8]) -> Option<Result<u16>> {
    if payload.len() != 22 || &payload[16..20] != b"MTU1" {
        return None;
    }
    Some(validate_mtu(u16::from_be_bytes(
        payload[20..22].try_into().unwrap(),
    )))
}

pub(crate) fn reply(request: &[u8], mtu: u16) -> Vec<u8> {
    let mut payload = request[..16].to_vec();
    payload.extend_from_slice(b"MTUA");
    payload.extend_from_slice(&mtu.to_be_bytes());
    payload
}

pub(crate) fn acknowledged(payload: &[u8]) -> Option<Result<u16>> {
    if payload.len() != 22 || &payload[16..20] != b"MTUA" {
        return None;
    }
    Some(validate_mtu(u16::from_be_bytes(
        payload[20..22].try_into().unwrap(),
    )))
}

pub(crate) fn confirm(payload: &[u8], challenge: &[u8; 16], offered: u16) -> Result<u16> {
    if payload.get(..16) != Some(challenge) {
        return Err(Error::Authentication(
            "invalid MTU confirmation challenge".into(),
        ));
    }
    if let Some(mtu) = acknowledged(payload) {
        let mtu = mtu?;
        if mtu <= offered {
            return Ok(mtu);
        }
        return Err(Error::Authentication("exit exceeded offered MTU".into()));
    }
    if payload == request(challenge, offered) && offered >= DEFAULT_INNER_MTU {
        return Ok(DEFAULT_INNER_MTU); // TG3 exit predating MTU negotiation.
    }
    Err(Error::HandshakeFailed(
        "Exit does not support the requested MTU".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encapsulation_budget_accounts_for_both_outer_ip_families() {
        assert_eq!(path_inner_mtu(1500, false).unwrap(), 1367);
        assert_eq!(path_inner_mtu(1500, true).unwrap(), 1359);
        assert_eq!(path_inner_mtu(1430, false).unwrap(), 1309);
        assert_eq!(path_inner_mtu(1430, true).unwrap(), 1289);
        assert_eq!(path_inner_mtu(1421, true).unwrap(), 1280);
        assert!(path_inner_mtu(1420, true).is_err());
        assert!(path_inner_mtu(0, false).is_err());
    }

    #[test]
    fn confirmation_binds_challenge_and_never_increases_offered_mtu() {
        let challenge = [0x51; 16];
        let probe = request(&challenge, 1320);
        assert_eq!(requested(&probe).unwrap().unwrap(), 1320);
        let response = reply(&probe, 1280);
        assert_eq!(confirm(&response, &challenge, 1320).unwrap(), 1280);
        assert!(confirm(&response, &[0; 16], 1320).is_err());
        assert!(confirm(&reply(&probe, 1340), &challenge, 1320).is_err());
        assert!(confirm(&reply(&probe, 1279), &challenge, 1320).is_err());
        assert!(confirm(&probe, &challenge, 1320).is_err());
        let compatible = request(&challenge, 1340);
        assert_eq!(confirm(&compatible, &challenge, 1340).unwrap(), 1340);
    }

    #[test]
    fn malformed_and_truncated_control_payloads_cannot_panic() {
        for size in 0..64 {
            let data = vec![0xff; size];
            assert!(requested(&data).is_none());
            assert!(acknowledged(&data).is_none());
            assert!(confirm(&data, &[0; 16], 1340).is_err());
        }
    }
}
