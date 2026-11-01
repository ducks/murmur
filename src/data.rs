/// Stable, explicitly specified PRNG so datasets do not depend on library versions.
#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}
/// Independent trials: cue 0 or 1, marker 2, target matching the cue.
/// The cue is sampled independently for each trial and train/test use disjoint streams.
pub fn trials(seed: u64, count: usize) -> Vec<usize> {
    let mut rng = Rng(seed);
    (0..count).map(|_| (rng.next_u64() & 1) as usize).collect()
}
