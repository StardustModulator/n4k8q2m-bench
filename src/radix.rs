//! Parallel LSD radix sort. 16-bit digits, stable, built for the dedup and
//! group passes. Scatter writes are disjoint, so threads do not lock.

use std::thread;

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T: Copy> Send for SendPtr<T> {}
unsafe impl<T: Copy> Sync for SendPtr<T> {}

impl<T: Copy> SendPtr<T> {
    unsafe fn read(self, index: usize) -> T {
        unsafe { self.0.add(index).read() }
    }
    unsafe fn write(self, index: usize, value: T) {
        unsafe { self.0.add(index).write(value) };
    }
}

pub fn par_radix_u64<T, F>(items: Vec<T>, threads: usize, key: F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> u64 + Sync,
{
    par_radix(items, threads, 4, &key)
}

pub fn par_radix_u32<T, F>(items: Vec<T>, threads: usize, key: F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> u32 + Sync,
{
    par_radix(items, threads, 2, &|item| key(item) as u64)
}

fn par_radix<T, F>(mut a: Vec<T>, threads: usize, passes: u32, key: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> u64 + Sync,
{
    let n = a.len();
    if n <= 1 {
        return a;
    }
    if n < 8_192 || threads <= 1 {
        a.sort_unstable_by_key(key);
        return a;
    }
    let threads = threads.max(1).min(n / 4_096).max(1);
    let mut b: Vec<T> = Vec::with_capacity(n);
    unsafe { b.set_len(n) };
    for pass in 0..passes {
        let shift = pass * 16;
        let mut counts = vec![vec![0u32; 65536]; threads];
        thread::scope(|scope| {
            for (t, count) in counts.iter_mut().enumerate() {
                let start = t * n / threads;
                let end = (t + 1) * n / threads;
                let src = &a[start..end];
                scope.spawn(move || {
                    for item in src {
                        let digit = ((key(item) >> shift) & 0xffff) as usize;
                        count[digit] += 1;
                    }
                });
            }
        });
        let mut sum = 0u32;
        for digit in 0..65536 {
            for count in counts.iter_mut() {
                let c = count[digit];
                count[digit] = sum;
                sum += c;
            }
        }
        debug_assert_eq!(sum as usize, n);
        let src = SendPtr(a.as_ptr() as *mut T);
        let dst = SendPtr(b.as_mut_ptr());
        thread::scope(|scope| {
            for (t, curs) in counts.iter_mut().enumerate() {
                let start = t * n / threads;
                let end = (t + 1) * n / threads;
                scope.spawn(move || {
                    for i in start..end {
                        let item = unsafe { src.read(i) };
                        let digit = ((key(&item) >> shift) & 0xffff) as usize;
                        let at = curs[digit] as usize;
                        curs[digit] += 1;
                        unsafe { dst.write(at, item) };
                    }
                });
            }
        });
        std::mem::swap(&mut a, &mut b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_big_sort() {
        let n = 8_000_000usize;
        let mut vals: Vec<(u64, u64)> = (0..n).map(|i| {
            let x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            (x, i as u64)
        }).collect();
        let t = std::time::Instant::now();
        vals = par_radix_u64(vals, 8, |v| v.0);
        eprintln!("radix 8M pairs {} ms", t.elapsed().as_millis());
        assert!(vals.windows(2).all(|w| w[0].0 <= w[1].0));
        let mut vals2: Vec<u64> = (0..n).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15)).collect();
        let t = std::time::Instant::now();
        vals2.sort_unstable();
        eprintln!("pdqsort 8M u64 {} ms", t.elapsed().as_millis());
    }

    #[test]
    fn sorts_u64_keys() {
        let mut vals: Vec<(u64, u32)> = (0..10_000u32).map(|i| (((i as u64) * 17 + 3) % 9_991, i)).collect();
        let expect = {
            let mut e = vals.clone();
            e.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            e
        };
        vals = par_radix_u64(vals, 4, |v| v.0);
        assert_eq!(vals, expect);
    }
}
