//! Matched study of two saved candidate updates across rule and phase lengths.
#[path = "../milestone_candidates/idea1.rs"] mod idea1;
#[path = "../milestone_candidates/idea2.rs"] mod idea2;
// The preserved candidates refer to the library's module through crate::.
pub mod unlabeled_context { pub use murmur_research_runner::unlabeled_context::*; }
use murmur::data::Rng;
use murmur_research_runner::{continual_rule_change::Markov, temporal_stream::StreamConfig, unlabeled_context::Mixture};
use std::{env, fs, path::Path};
const NAMES: [&str; 6] = ["single", "mixture", "idea1_predictive_gate", "idea2_posterior_1_25", "posterior_2", "oracle"];
const FAMILIES: [(&str, [[f64; 2]; 2]); 5] = [
    ("no_change", [[0.20,0.80],[0.20,0.80]]),
    ("subtle_shift", [[0.20,0.80],[0.30,0.70]]),
    ("high_noise", [[0.40,0.60],[0.60,0.40]]),
    ("asymmetric_new", [[0.10,0.65],[0.85,0.45]]),
    ("uneven_bias", [[0.05,0.15],[0.70,0.85]]),
];
const LENGTHS: [usize; 3] = [100,400,1000];
#[derive(Clone, Copy, Debug, PartialEq)]
struct Event { x: usize, y: usize, context: usize }
fn events(seed:u64, contexts:&[usize], rules:[[f64;2];2]) -> Vec<Event> {
    let mut rng=Rng(seed); let mut x=(rng.next_u64() & 1) as usize;
    contexts.iter().map(|&context| {let y=usize::from(rng.uniform()<rules[context][x]); let e=Event{x,y,context}; x=y; e}).collect()
}
#[derive(Clone)]
struct Models { single:Markov, mixture:Mixture, idea1:idea1::Candidate, idea2:idea2::Candidate, idea2_inspect:Mixture, square:Mixture, oracle:Markov }
impl Models {
    fn new(seed:u64, rate:f64)->Self { Self {single:Markov::new(rate,false),mixture:Mixture::new(seed,rate),
        idea1:idea1::Candidate::new(seed,rate),idea2:idea2::Candidate::new(seed,rate),idea2_inspect:Mixture::new(seed,rate),
        square:Mixture::new(seed,rate),oracle:Markov::new(rate,true)} }
    fn reset(&mut self) {self.mixture.reset();self.idea1.reset();self.idea2.reset();self.idea2_inspect.reset();self.square.reset();}
    fn step(&mut self,e:Event,learn:bool)->[f64;6] {
        let first=self.single.p[0][e.x];
        let mix=self.mixture.step(e.x,e.y,learn).0;
        let a=self.idea1.predict(e.x);let b=self.idea2.predict(e.x);
        let inspect_prediction=self.idea2_inspect.step(e.x,e.y,false).0;
        assert!((b-inspect_prediction).abs()<1e-12, "inspection copy diverged from the saved candidate");
        let square=self.square.step(e.x,e.y,false).0;
        let oracle=self.oracle.p[e.context][e.x];
        let p=[first,mix,a,b,square,oracle];
        assert!(p.iter().all(|x|x.is_finite()&&(0.0..=1.0).contains(x)));
        self.idea1.observe(e.x,e.y,learn);self.idea2.observe(e.x,e.y,learn);
        if learn {
            for k in 0..2 { let posterior=self.idea2_inspect.belief[k]; let weight=posterior*posterior.sqrt().sqrt();
                self.idea2_inspect.p[k][e.x]=(self.idea2_inspect.p[k][e.x]+self.idea2_inspect.rate*weight*(e.y as f64-self.idea2_inspect.p[k][e.x])).clamp(0.001,0.999); }
            for k in 0..2 { let weight=self.square.belief[k]*self.square.belief[k];
                self.square.p[k][e.x]=(self.square.p[k][e.x]+self.square.rate*weight*(e.y as f64-self.square.p[k][e.x])).clamp(0.001,0.999); }
            self.single.learn(e.x,e.y,e.context);self.oracle.learn(e.x,e.y,e.context);
        }
        p
    }
}
#[derive(Default)]
struct Scores {correct:[usize;6], loss:[f64;6], count:usize}
impl Scores {
    fn add(&mut self,p:[f64;6],y:usize) {self.count+=1;for k in 0..6 {
        self.correct[k]+=usize::from(usize::from(p[k]>0.5)==y);
        self.loss[k]-=if y==1 {p[k]} else {1.0-p[k]}.clamp(1e-12,1.0).ln();
    }}
    fn rows(&self,out:&mut String,seed:u64,family:&str,length:usize,phase:usize,kind:&str) {
        for (k,name) in NAMES.iter().enumerate() {out.push_str(&format!("{seed},{family},{length},{phase},{kind},{name},{},{},{:.8},{:.8}\n",self.correct[k],self.count,self.correct[k] as f64/self.count as f64,self.loss[k]/self.count as f64));}
    }
}
fn run(c:&StreamConfig,seed:u64,family:&str,rules:[[f64;2];2],length:usize,out:&mut String,states:&mut String) {
    let contexts:Vec<_>=(0..3*length).map(|t|usize::from(t>=length&&t<2*length)).collect();
    let train=events(seed.wrapping_add(40000),&contexts,rules);
    let probes=[events(seed.wrapping_add(46000),&vec![0;c.test_tokens],rules),events(seed.wrapping_add(48000),&vec![1;c.test_tokens],rules)];
    let mut model=Models::new(seed.wrapping_add(51000),c.learning_rate);
    for phase in 0..3 {
        let mut online=Scores::default();let mut first=Scores::default();
        for (i,&e) in train[phase*length..(phase+1)*length].iter().enumerate() {
            let p=model.step(e,true);online.add(p,e.y);if i<100 {first.add(p,e.y);}
        }
        online.rows(out,seed,family,length,phase+1,"online");first.rows(out,seed,family,length,phase+1,"first_100");
        for (name,m) in [("mixture",&model.mixture),("idea2_inspect",&model.idea2_inspect),("posterior_2",&model.square)] {
            states.push_str(&format!("{seed},{family},{length},{},{name},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}\n",phase+1,m.p[0][0],m.p[0][1],m.p[1][0],m.p[1][1],m.belief[0],m.belief[1]));
        }
        for (k,probe) in probes.iter().enumerate() {let mut frozen=model.clone();frozen.reset();let mut score=Scores::default();
            for &e in probe {score.add(frozen.step(e,false),e.y);}
            score.rows(out,seed,family,length,phase+1,if k==0 {"probe_a"} else {"probe_b"});
        }
    }
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=env::args().skip(1).collect();if a.len()!=2 {return Err("usage: update_rule_sweep <config> <new-output-directory>".into());}
    let c=StreamConfig::parse(&fs::read_to_string(&a[0])?)?;let dir=Path::new(&a[1]);fs::create_dir(dir)?;
    let mut csv=String::from("seed,family,phase_length,phase,measurement,model,correct,trials,accuracy,logloss\n");
    let mut states=String::from("seed,family,phase_length,phase,model,p00,p01,p10,p11,belief0,belief1\n");
    for &seed in &c.seeds {for (family,rules) in FAMILIES {for length in LENGTHS {run(&c,seed,family,rules,length,&mut csv,&mut states);}}}
    fs::write(dir.join("metrics.csv"),csv)?;fs::write(dir.join("states.csv"),states)?;Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn prediction_is_before_target_and_frozen_learning_stays_frozen() {
        let mut a=Models::new(17,0.05);let mut b=a.clone();
        assert_eq!(a.step(Event{x:0,y:0,context:0},false),b.step(Event{x:0,y:1,context:0},false));
        a.reset();b.reset();for x in 0..2 {assert_eq!(a.idea1.predict(x),b.idea1.predict(x));assert_eq!(a.idea2.predict(x),b.idea2.predict(x));assert_eq!(a.square.p,b.square.p);}
    }
    #[test] fn streams_are_paired_and_deterministic() {
        let seq=[0,0,1,1];let rules=[[0.0,0.0],[1.0,1.0]];
        let a=events(77,&seq,rules);assert_eq!(a,events(77,&seq,rules));
        for (e,c) in a.iter().zip(seq) {assert_eq!(e.y,c);}
        for w in a.windows(2) {assert_eq!(w[0].y,w[1].x);}
    }
    #[test] fn square_variant_changes_only_the_update_weight() {
        let mut m=Models::new(47,0.05);let before=m.square.clone();
        let event=Event{x:0,y:1,context:0};let prior_prediction=m.square.clone().step(0,1,false).0;
        assert_eq!(m.step(event,true)[4],prior_prediction);
        let mut reference=before;reference.step(0,1,false);
        for k in 0..2 {let want=(reference.p[k][0]+reference.rate*reference.belief[k].powi(2)*(1.0-reference.p[k][0])).clamp(0.001,0.999);
            assert!((m.square.p[k][0]-want).abs()<1e-12);}
    }
    #[test] fn idea2_matches_baseline_with_posterior_power_1_25() {
        let mut model=idea2::Candidate::new(47,0.05);
        let mut reference=Mixture::new(47,0.05);
        for e in events(81,&vec![0;1000],FAMILIES[3].1) {
            let predicted=model.predict(e.x);
            let expected=reference.step(e.x,e.y,false).0;
            assert!((predicted-expected).abs()<1e-12);
            model.observe(e.x,e.y,true);
            for k in 0..2 {
                let w=reference.belief[k]*reference.belief[k].sqrt().sqrt();
                reference.p[k][e.x]=(reference.p[k][e.x]+reference.rate*w*(e.y as f64-reference.p[k][e.x])).clamp(0.001,0.999);
            }
        }
    }
}
