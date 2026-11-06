//! Independent paired evaluation of a saved candidate and its correction controls.
#[path="../validated_candidates/seeded_correction.rs"] pub mod candidate;
pub mod unlabeled_context { pub use murmur_research_runner::unlabeled_context::*; }
use murmur::data::Rng;
use murmur_research_runner::{temporal_stream::StreamConfig,unlabeled_context::Mixture};
use std::{env,fs,path::Path};
const FAMILIES:[(&str,[[f64;2];2]);6]=[("uneven_bias",[[0.05,0.15],[0.70,0.85]]),("shifted_bias",[[0.10,0.20],[0.65,0.90]]),("similar_rules",[[0.10,0.70],[0.30,0.90]]),("high_entropy",[[0.40,0.60],[0.60,0.40]]),("reverse_asymmetric",[[0.80,0.30],[0.20,0.75]]),("stationary",[[0.20,0.80],[0.20,0.80]])];
const ARMS:[&str;4]=["no_correction","guarded","unguarded","mixture"];
const LENGTHS:[usize;2]=[400,1000];
fn stream(seed:u64,contexts:&[usize],rules:[[f64;2];2])->Vec<(usize,usize)> {let mut r=Rng(seed);let mut x=(r.next_u64()&1) as usize;contexts.iter().map(|&c|{let y=usize::from(r.uniform()<rules[c][x]);let e=(x,y);x=y;e}).collect()}
#[derive(Clone)]
struct Models {candidate:[candidate::Candidate;3],mixture:Mixture}
impl Models {
 fn new(seed:u64,rate:f64)->Self {let a=candidate::Candidate::new(seed,rate);Self{candidate:[a.clone(),a.guarded_smoothing(),a.unguarded_smoothing()],mixture:Mixture::new(seed,rate)}}
 fn reset(&mut self){for m in &mut self.candidate {m.reset();}self.mixture.reset();}
 fn step(&mut self,x:usize,y:usize,learn:bool)->[f64;4] {
  let p=[self.candidate[0].predict(x),self.candidate[1].predict(x),self.candidate[2].predict(x),self.mixture.step(x,y,learn).0];
  assert!(p.iter().all(|x|x.is_finite()&&(0.0..=1.0).contains(x)));for m in &mut self.candidate{m.observe(x,y,learn);}p
 }
}
#[derive(Default)] struct Scores{correct:[usize;4],loss:[f64;4],n:usize}
impl Scores {
 fn add(&mut self,p:[f64;4],y:usize){self.n+=1;for k in 0..4 {self.correct[k]+=usize::from(usize::from(p[k]>0.5)==y);self.loss[k]-=if y==1{p[k]}else{1.0-p[k]}.clamp(1e-12,1.0).ln();}}
 fn write(&self,out:&mut String,seed:u64,f:&str,len:usize,phase:usize,metric:&str){for(k,name)in ARMS.iter().enumerate(){out.push_str(&format!("{seed},{f},{len},{phase},{metric},{name},{},{},{:.8},{:.8}\n",self.correct[k],self.n,self.correct[k]as f64/self.n as f64,self.loss[k]/self.n as f64));}}
}
fn run(c:&StreamConfig,seed:u64,f:&str,rules:[[f64;2];2],len:usize,out:&mut String,diagnostics:&mut String){
 let contexts:Vec<_>=(0..3*len).map(|t|usize::from(t>=len&&t<2*len)).collect();let data=stream(seed.wrapping_add(40000),&contexts,rules);
 let probes=[stream(seed.wrapping_add(46000),&vec![0;c.test_tokens],rules),stream(seed.wrapping_add(48000),&vec![1;c.test_tokens],rules)];
 let mut m=Models::new(seed.wrapping_add(51000),c.learning_rate);
 for phase in 0..3{let mut online=Scores::default();let mut first=Scores::default();let mut sums=[[0.0;3];2];
  for(i,&(x,y))in data[phase*len..(phase+1)*len].iter().enumerate(){let p=m.step(x,y,true);online.add(p,y);if i<100{first.add(p,y);}
   if let Some((guard,magnitude))=m.candidate[1].correction_diagnostic(){let bucket=usize::from(i>=100);sums[bucket][0]+=guard;sums[bucket][1]+=magnitude;sums[bucket][2]+=1.0;}}
  online.write(out,seed,f,len,phase+1,"online");first.write(out,seed,f,len,phase+1,"first_100");
  for (label,values) in [("first_100",sums[0]),("remaining",sums[1])]{if values[2]>0.0{diagnostics.push_str(&format!("{seed},{f},{len},{},{label},{:.12},{:.12},{}\n",phase+1,values[0]/values[2],values[1]/values[2],values[2] as usize));}}
  for(k,probe)in probes.iter().enumerate(){let mut frozen=m.clone();frozen.reset();let mut score=Scores::default();for &(x,y)in probe{score.add(frozen.step(x,y,false),y);}score.write(out,seed,f,len,phase+1,if k==0{"probe_a"}else{"probe_b"});}
 }
}
fn main()->Result<(),Box<dyn std::error::Error>>{let a:Vec<_>=env::args().skip(1).collect();if a.len()!=2{return Err("usage: correction_validation <config> <new-output-directory>".into());}let c=StreamConfig::parse(&fs::read_to_string(&a[0])?)?;let dir=Path::new(&a[1]);fs::create_dir(dir)?;
 let mut out=String::from("seed,family,phase_length,phase,measurement,model,correct,trials,accuracy,logloss\n");let mut diagnostics=String::from("seed,family,phase_length,phase,window,mean_guard,mean_abs_correction,corrections\n");for &seed in &c.seeds{for(f,rules)in FAMILIES{for len in LENGTHS{run(&c,seed,f,rules,len,&mut out,&mut diagnostics);}}}fs::write(dir.join("metrics.csv"),out)?;fs::write(dir.join("guard-diagnostics.csv"),diagnostics)?;Ok(())}
#[cfg(test)] mod tests{use super::*;
 #[test]fn predictions_precede_targets(){let mut a=Models::new(10,0.05);let mut b=a.clone();assert_eq!(a.step(0,0,true),b.step(0,1,true));}
 #[test]fn frozen_observations_leave_parameters_unchanged_after_reset(){let mut a=Models::new(10,0.05);for i in 0..100{a.step(i%2,(i/3)%2,true);}let mut b=a.clone();for i in 0..100{b.step(i%2,(i/7)%2,false);}a.reset();b.reset();for x in 0..2 {assert_eq!(a.candidate[0].predict(x),b.candidate[0].predict(x));assert_eq!(a.candidate[1].predict(x),b.candidate[1].predict(x));assert_eq!(a.candidate[2].predict(x),b.candidate[2].predict(x));assert_eq!(a.mixture.p,b.mixture.p);}}
 #[test]fn seeded_stream_is_deterministic(){let rules=[[0.0,0.0],[1.0,1.0]];let contexts=[0,1,0,1];let x=stream(4,&contexts,rules);assert_eq!(x,stream(4,&contexts,rules));for (e,c) in x.iter().zip(contexts){assert_eq!(e.1,c);}}
}
