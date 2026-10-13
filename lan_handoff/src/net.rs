//! LAN の IPv4 アドレス検出とクライアント許可判定。

use std::net::Ipv4Addr;

pub struct Lan {
    pub name: String,
    pub ip: Ipv4Addr,
    pub mask: Ipv4Addr,
}

/// プライベート IPv4(192.168/16, 10/8, 172.16/12)を持つインターフェースの候補。
/// ループバックとリンクローカルは除く。
pub fn lan_candidates() -> Vec<Lan> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4)
                if !v4.ip.is_loopback() && !v4.ip.is_link_local() && v4.ip.is_private() =>
            {
                Some(Lan {
                    name: i.name,
                    ip: v4.ip,
                    mask: v4.netmask,
                })
            }
            _ => None,
        })
        .collect()
}

/// `ip` が属するインターフェースのネットマスク。見つからなければ /32 とみなす(同一ホストのみ)。
pub fn mask_for(ip: Ipv4Addr) -> Ipv4Addr {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .find_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4) if v4.ip == ip => Some(v4.netmask),
            _ => None,
        })
        .unwrap_or(Ipv4Addr::new(255, 255, 255, 255))
}

pub fn same_subnet(a: Ipv4Addr, b: Ipv4Addr, mask: Ipv4Addr) -> bool {
    let m = u32::from(mask);
    u32::from(a) & m == u32::from(b) & m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subnet_check() {
        let mask = Ipv4Addr::new(255, 255, 255, 0);
        let me = Ipv4Addr::new(192, 168, 11, 3);
        assert!(same_subnet(Ipv4Addr::new(192, 168, 11, 40), me, mask));
        assert!(!same_subnet(Ipv4Addr::new(192, 168, 12, 40), me, mask));
        assert!(!same_subnet(Ipv4Addr::new(10, 0, 0, 1), me, mask));
    }
}
