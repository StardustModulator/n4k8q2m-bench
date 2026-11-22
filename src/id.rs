//! Pack generator-style ids (`u`/`e`/`k` + base36, or the fully upper-cased
//! copy) into a `u64`. Anything else is a slow string.

/// Bit 62 marks a slow-string id. Packed ids never set it: the prefix lives in
/// bits 48..56 and the case flag in bit 63.
pub const SLOW_TAG: u64 = 1 << 62;

#[inline]
pub fn is_slow(key: u64) -> bool {
    key & SLOW_TAG != 0
}

#[inline]
pub fn slow_index(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

#[inline]
pub fn slow_key(index: u32) -> u64 {
    SLOW_TAG | index as u64
}

/// Base36 digit, or `0xFF` when the byte is not a digit or letter.
const B36: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut c = b'0';
    while c <= b'9' {
        t[c as usize] = c - b'0';
        c += 1;
    }
    let mut i = 0u8;
    while i < 26 {
        t[(b'a' + i) as usize] = 10 + i;
        t[(b'A' + i) as usize] = 10 + i;
        i += 1;
    }
    t
};

/// `None` when the text is not a canonical all-lower or all-upper base36 id.
#[inline(always)]
pub fn try_pack(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n < 2 || n > 14 {
        return None;
    }
    let first = s[0];
    let upper = first.wrapping_sub(b'A') <= 25;
    let lower = first.wrapping_sub(b'a') <= 25;
    if upper == lower {
        return None;
    }
    if n > 2 && s[1] == b'0' {
        return None;
    }
    let mut value: u64 = 0;
    let mut i = 1;
    while i < n {
        let c = s[i];
        let digit = B36[c as usize];
        if digit == 0xFF {
            return None;
        }
        if digit >= 10 && (c >= b'a') == upper {
            return None;
        }
        value = value * 36 + digit as u64;
        i += 1;
    }
    if value >= (1 << 48) {
        return None;
    }
    let prefix = if upper { first + 32 } else { first } as u64;
    let case_bit = if upper { 1u64 << 63 } else { 0 };
    Some(case_bit | (prefix << 48) | value)
}

/// Write the canonical text of a packed id. Returns the length.
#[inline]
pub fn write_id(id: u64, buf: &mut [u8; 24]) -> usize {
    debug_assert!(!is_slow(id));
    let upper = id & (1 << 63) != 0;
    let prefix = ((id >> 48) & 0xFF) as u8;
    let mut value = id & ((1 << 48) - 1);
    let mut tmp = [0u8; 16];
    let mut n = 0usize;
    if value == 0 {
        tmp[0] = b'0';
        n = 1;
    } else {
        while value > 0 {
            let digit = (value % 36) as u8;
            tmp[n] = if digit < 10 { b'0' + digit } else { b'a' + (digit - 10) };
            n += 1;
            value /= 36;
        }
    }
    buf[0] = prefix;
    for i in 0..n {
        buf[1 + i] = tmp[n - 1 - i];
    }
    let len = n + 1;
    if upper {
        for b in &mut buf[..len] {
            b.make_ascii_uppercase();
        }
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_generator_ids() {
        for sample in [
            "uduffvac",
            "u8w84o3t0",
            "U14NPKDLL",
            "e42huhoug",
            "ec1k0erc1",
            "kc1ni90tn",
            "k1",
            "u0",
            "e0",
            "U0",
        ] {
            let packed = try_pack(sample.as_bytes()).unwrap_or_else(|| panic!("{sample}"));
            let mut buf = [0u8; 24];
            let n = write_id(packed, &mut buf);
            assert_eq!(&buf[..n], sample.as_bytes(), "{sample}");
        }
        assert!(try_pack(b"u").is_none());
        assert!(try_pack(b"u01").is_none());
        assert!(try_pack(b"uA").is_none());
        assert!(try_pack(b" u1").is_none());
        assert!(try_pack(b"").is_none());
    }
}
