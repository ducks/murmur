//! Causal diagnostic: restore one saved slot after A returns, then probe.
#[path="../milestone_candidates/idea2.rs"] mod idea2;
pub mod unlabeled_context { pub use murmur_research_runner::unlabeled_context::*; }
use murmur::data::Rng;
use murmur_research_runner::{temporal_stream::StreamConfig,unlabeled_context::Mixture};
use std::{env,fs,path::Path};
const FAMILIES:[(&str,[[f64;2];2]);2]=[("uneven_bias",[[0.05,0.15],[0.70,0.85]]),("shifted_bias",[[0.10,0.20],[0.65,0.90]])];
fn stream(seed:u64,contexts:&[usize],rules:[[f64;2];2])->Vec<(usize,usize)> {
 let mut r=Rng(seed);let mut x=(r.next_u64()&1) as usize;
 contexts.iter().map(|&c|{let y=usize::from(r.uniform()<rules[c][x]);let e=(x,y);x=y;e}).collect()
}
fn observe(m:&mut Mixture,x:usize,y:usize,learn:bool)->f64 {
 let p=m.step(x,y,false).0;
 if learn {for k in 0..2 {let b=m.belief[k];let w=b*b.sqrt().sqrt();m.p[k][x]=(m.p[k][x]+m.rate*w*(y as f64-m.p[k][x])).clamp(0.001,0.999);}}
 p
}
fn accuracy(mut m:Mixture,data:&[(usize,usize)],weighted:bool)->usize {
 m.reset();data.iter().map(|&(x,y)| {let p=if weighted {observe(&mut m,x,y,false)} else {m.step(x,y,false).0};usize::from(usize::from(p>0.5)==y)}).sum()
}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=env::args().skip(1).collect();if a.len()!=2 {return Err("usage: memory_intervention <config> <new-output-directory>".into());}
 let c=StreamConfig::parse(&fs::read_to_string(&a[0])?)?;let d=Path::new(&a[1]);fs::create_dir(d)?;
 let mut out=String::from("seed,family,arm,probe,correct,trials,accuracy,selected_slot\n");
 for &seed in &c.seeds {for (family,rules) in FAMILIES {
  let n=c.train_tokens;let contexts:Vec<_>=(0..3*n).map(|t|usize::from(t>=n&&t<2*n)).collect();let data=stream(seed.wrapping_add(40000),&contexts,rules);
  let probes=[stream(seed.wrapping_add(46000),&vec![0;c.test_tokens],rules),stream(seed.wrapping_add(48000),&vec![1;c.test_tokens],rules)];
  let mut model=Mixture::new(seed.wrapping_add(51000),c.learning_rate);let mut original=idea2::Candidate::new(seed.wrapping_add(51000),c.learning_rate);let mut baseline=Mixture::new(seed.wrapping_add(51000),c.learning_rate);let mut saved=[[0.0;2];2];let mut selected=0;
  for (i,&(x,y)) in data.iter().enumerate() {
   let p=observe(&mut model,x,y,true);assert!((p-original.predict(x)).abs()<1e-12);original.observe(x,y,true);baseline.step(x,y,true);
   if i+1==2*n {saved=model.p;selected=usize::from(saved[1][0]+saved[1][1]>saved[0][0]+saved[0][1]);}
  }
  let sham=model.clone();let mut selected_restore=model.clone();selected_restore.p[selected]=saved[selected];let mut other_restore=model.clone();other_restore.p[1-selected]=saved[1-selected];
  for (arm,m,weighted) in [("sham",sham,true),("restore_selected",selected_restore,true),("restore_other",other_restore,true),("mixture",baseline,false)] {
   for (rule,probe) in probes.iter().enumerate() {let correct=accuracy(m.clone(),probe,weighted);out.push_str(&format!("{seed},{family},{arm},{},{correct},{},{:.8},{selected}\n",if rule==0 {"a"}else{"b"},probe.len(),correct as f64/probe.len() as f64));}
  }
 }}
 fs::write(d.join("metrics.csv"),out)?;Ok(())
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn stream_is_deterministic_and_respects_rules() {let rules=[[0.0,0.0],[1.0,1.0]];let c=[0,1,0];let x=stream(4,&c,rules);assert_eq!(x,stream(4,&c,rules));for (e,c) in x.iter().zip(c){assert_eq!(e.1,c);}}
 #[test] fn frozen_observation_keeps_parameters() {let mut m=Mixture::new(1,0.05);let p=m.p;for _ in 0..50{observe(&mut m,0,1,false);}assert_eq!(p,m.p);}
}
