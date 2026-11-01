use crate::data::Rng;

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}
#[derive(Clone)]
pub struct Graph {
    pub state: Vec<f64>,
    pub edges: Vec<Edge>,
    pub input: Vec<Vec<f64>>,
    pub leak: f64,
}
impl Graph {
    pub fn new(n: usize, degree: usize, seed: u64) -> Self {
        Self::with_symbols(n, degree, seed, 3)
    }
    pub fn with_symbols(n: usize, degree: usize, seed: u64, symbols: usize) -> Self {
        assert!(n > 0 && degree > 0);
        assert!(symbols > 0);
        let mut rng = Rng(seed);
        let mut edges = Vec::new();
        for to in 0..n {
            for _ in 0..degree {
                edges.push(Edge {
                    from: (rng.next_u64() as usize) % n,
                    to,
                    weight: (rng.uniform() * 2.0 - 1.0) * 0.5 / (degree as f64).sqrt(),
                });
            }
        }
        let input = (0..n)
            .map(|_| (0..symbols).map(|_| rng.uniform() * 2.0 - 1.0).collect())
            .collect();
        Self {
            state: vec![0.0; n],
            edges,
            input,
            leak: 0.35,
        }
    }
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
    /// Synchronous update: all messages use the previous state, never partially updated state.
    pub fn step(&mut self, symbol: usize) {
        assert!(symbol < self.input[0].len());
        let mut incoming: Vec<f64> = self.input.iter().map(|w| w[symbol]).collect();
        for edge in &self.edges {
            incoming[edge.to] += edge.weight * self.state[edge.from];
        }
        for (s, x) in self.state.iter_mut().zip(incoming) {
            *s = (1.0 - self.leak) * *s + self.leak * x.tanh();
        }
    }
}
