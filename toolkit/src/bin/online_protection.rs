//! Online slot protection with an independent random-slot control.
#[path="../milestone_candidates/idea2.rs"] mod idea2;
pub mod unlabeled_context { pub use murmur_research_runner::unlabeled_context::*; }
use murmur::data::Rng;
use murmur_research_runner::{temporal_stream::StreamConfig,unlabeled_context::Mixture};
use std::{env,fs,path::Path};
const FAMILIES:[(&str,[[f64;2];2]);4]=[("uneven_bias",[[0.05,0.15],[0.70,0.85]]),("shifted_bias",[[0.10,0.20],[0.65,0.90]]),("complementary",[[0.25,0.75],[0.75,0.25]]),("no_change",[[0.20,0.80],[0.20,0.80]])];
const ARMS:[&str;4]=["mixture","idea2","protect_inactive","random_slot_control"];
const LENGTHS:[usize;2]=[400,1000];
fn stream(seed:u64,contexts:&[usize],rules:[[f64;2];2])->Vec<(usize,usize)> {let mut r=Rng(seed);let mut x=(r.next_u64()&1) as usize;contexts.iter().map(|&c|{let y=usize::from(r.uniform()<rules[c][x]);let e=(x,y);x=y;e}).collect()}
#[derive(Clone)]
struct Policy { m:Mixture, rng:Rng, random:bool, opportunities:usize, suppressed:usize }
impl Policy {
 fn new(seed:u64,rng_seed:u64,rate:f64,random:bool)->Self {Self{m:Mixture::new(seed,rate),rng:Rng(rng_seed),random,opportunities:0,suppressed:0}}
 fn reset(&mut self){self.m.reset();}
 fn step(&mut self,x:usize,y:usize,learn:bool)->f64 {
  let (p,prior,_)=self.m.step(x,y,false);
  if learn {
   let active=usize::from(prior[1]>prior[0]);let confident=prior[active]>0.8;
   let frozen=if confident {self.opportunities+=1;self.suppressed+=1;Some(if self.random {(self.rng.next_u64()&1) as usize}else{1-active})}else{None};
   for k in 0..2 {if frozen==Some(k){continue;}let b=self.m.belief[k];let w=b*b.sqrt().sqrt();self.m.p[k][x]=(self.m.p[k][x]+self.m.rate*w*(y as f64-self.m.p[k][x])).clamp(0.001,0.999);}
  }
  p
 }
}
#[derive(Clone)]
struct Models { baseline:Mixture, candidate:idea2::Candidate, gate:Policy, random:Policy }
impl Models {
 fn new(seed:u64,control_seed:u64,rate:f64)->Self {Self{baseline:Mixture::new(seed,rate),candidate:idea2::Candidate::new(seed,rate),gate:Policy::new(seed,control_seed,rate,false),random:Policy::new(seed,control_seed,rate,true)}}
 fn reset(&mut self){self.baseline.reset();self.candidate.reset();self.gate.reset();self.random.reset();}
 fn step(&mut self,x:usize,y:usize,learn:bool)->[f64;4] {let a=self.baseline.step(x,y,learn).0;let b=self.candidate.predict(x);self.candidate.observe(x,y,learn);let c=self.gate.step(x,y,learn);let d=self.random.step(x,y,learn);let p=[a,b,c,d];assert!(p.iter().all(|x|x.is_finite()&&(0.0..=1.0).contains(x)));p}
}
#[derive(Default)] struct Scores {correct:[usize;4],loss:[f64;4],n:usize}
impl Scores {
 fn add(&mut self,p:[f64;4],y:usize){self.n+=1;for k in 0..4 {self.correct[k]+=usize::from(usize::from(p[k]>0.5)==y);self.loss[k]-=if y==1{p[k]}else{1.0-p[k]}.clamp(1e-12,1.0).ln();}}
 fn write(&self,out:&mut String,seed:u64,f:&str,len:usize,phase:usize,metric:&str){for (k,name) in ARMS.iter().enumerate(){out.push_str(&format!("{seed},{f},{len},{phase},{metric},{name},{},{},{:.8},{:.8}\n",self.correct[k],self.n,self.correct[k] as f64/self.n as f64,self.loss[k]/self.n as f64));}}
}
fn run(c:&StreamConfig,seed:u64,f:&str,rules:[[f64;2];2],len:usize,out:&mut String,counts:&mut String){
 let contexts:Vec<_>=(0..3*len).map(|t|usize::from(t>=len&&t<2*len)).collect();let data=stream(seed.wrapping_add(40000),&contexts,rules);
 let probes=[stream(seed.wrapping_add(46000),&vec![0;c.test_tokens],rules),stream(seed.wrapping_add(48000),&vec![1;c.test_tokens],rules)];
 let mut m=Models::new(seed.wrapping_add(51000),seed.wrapping_add(71000),c.learning_rate);
 for phase in 0..3 {let mut online=Scores::default();let mut first=Scores::default();for (i,&(x,y)) in data[phase*len..(phase+1)*len].iter().enumerate(){let p=m.step(x,y,true);online.add(p,y);if i<100{first.add(p,y);}}
  online.write(out,seed,f,len,phase+1,"online");first.write(out,seed,f,len,phase+1,"first_100");
  for (k,probe) in probes.iter().enumerate(){let mut frozen=m.clone();frozen.reset();let mut score=Scores::default();for &(x,y) in probe{score.add(frozen.step(x,y,false),y);}score.write(out,seed,f,len,phase+1,if k==0{"probe_a"}else{"probe_b"});}
 }
 for (name,p) in [("protect_inactive",&m.gate),("random_slot_control",&m.random)] {counts.push_str(&format!("{seed},{f},{len},{name},{},{},{}\n",p.opportunities,p.suppressed,3*len));}
}
fn main()->Result<(),Box<dyn std::error::Error>>{let a:Vec<_>=env::args().skip(1).collect();if a.len()!=2{return Err("usage: online_protection <config> <new-output-directory>".into());}let c=StreamConfig::parse(&fs::read_to_string(&a[0])?)?;let dir=Path::new(&a[1]);fs::create_dir(dir)?;
 let mut out=String::from("seed,family,phase_length,phase,measurement,model,correct,trials,accuracy,logloss\n");let mut counts=String::from("seed,family,phase_length,model,eligible_steps,suppressed_updates,training_steps\n");
 for &seed in &c.seeds{for(f,rules) in FAMILIES{for len in LENGTHS{run(&c,seed,f,rules,len,&mut out,&mut counts);}}}fs::write(dir.join("metrics.csv"),out)?;fs::write(dir.join("suppression.csv"),counts)?;Ok(())}
#[cfg(test)] mod tests {use super::*;
 #[test] fn frozen_probes_do_not_change_parameters_or_rng(){let mut p=Policy::new(8,9,0.05,true);let before=p.m.p;let r=p.rng.0;for _ in 0..100{p.step(0,1,false);}assert_eq!(p.m.p,before);assert_eq!(p.rng.0,r);assert_eq!(p.suppressed,0);}
 #[test] fn high_confidence_protects_only_less_likely_slot(){let mut p=Policy::new(8,9,0.05,false);p.m.belief=[1.0,0.0];let before=p.m.p;p.step(0,1,true);assert_eq!(before[1],p.m.p[1]);assert_ne!(before[0][0],p.m.p[0][0]);assert_eq!(p.suppressed,1);}
 #[test] fn prediction_precedes_target(){let mut a=Models::new(1,2,0.05);let mut b=a.clone();assert_eq!(a.step(0,0,true),b.step(0,1,true));}
}
