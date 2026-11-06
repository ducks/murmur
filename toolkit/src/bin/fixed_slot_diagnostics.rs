//! Paired correction ablation with one fixed, oracle-assisted slot mapping.
#[path="../validated_candidates/diagnostic_candidate.rs"] pub mod candidate;
pub mod unlabeled_context { pub use murmur_research_runner::unlabeled_context::*; }
use murmur::data::Rng;
use murmur_research_runner::{temporal_stream::StreamConfig,unlabeled_context::{Mixture,HAZARD}};
use std::{env,fs,path::Path};
const FAMILIES:[(&str,[[f64;2];2]);4]=[("inverse_bias",[[0.85,0.90],[0.10,0.25]]),("moderate_noise",[[0.35,0.65],[0.65,0.35]]),("new_asymmetric",[[0.17,0.70],[0.80,0.35]]),("new_constant",[[0.12,0.12],[0.82,0.82]])];
const ARMS:[&str;3]=["correction","no_correction","mixture"];
const LENGTHS:[usize;2]=[400,1000];
fn stream(seed:u64,contexts:&[usize],rules:[[f64;2];2])->Vec<(usize,usize)> {let mut r=Rng(seed);let mut x=(r.next_u64()&1) as usize;contexts.iter().map(|&c|{let y=usize::from(r.uniform()<rules[c][x]);let e=(x,y);x=y;e}).collect()}
#[derive(Clone)]struct Models{c:[candidate::Candidate;2],m:Mixture}
impl Models{
 fn new(seed:u64,rate:f64)->Self{let a=candidate::Candidate::new(seed,rate);Self{c:[a.clone(),a.without_smoothing()],m:Mixture::new(seed,rate)}}
 fn reset(&mut self){for m in &mut self.c{m.reset();}self.m.reset();}
 fn state(&self)->[([f64;2],[[f64;2];2]);3]{[self.c[0].diagnostic_state(),self.c[1].diagnostic_state(),(self.m.belief,[[self.m.p[0][0],self.m.p[1][0]],[self.m.p[0][1],self.m.p[1][1]]])]}
 fn step(&mut self,x:usize,y:usize,learn:bool)->[f64;3]{let p=[self.c[0].predict(x),self.c[1].predict(x),self.m.step(x,y,learn).0];assert!(p.iter().all(|x|x.is_finite()&&(0.0..=1.0).contains(x)));for m in &mut self.c{m.observe(x,y,learn);}p}
}
fn align_a(rows:[[f64;2];2],rule:[f64;2])->(usize,f64){let e=|k:usize|(rows[0][k]-rule[0]).powi(2)+(rows[1][k]-rule[1]).powi(2);let a=e(0);let b=e(1);(usize::from(b<a),(a-b).abs())}
#[derive(Default)]struct Scores{correct:[usize;3],loss:[f64;3],n:usize}
impl Scores{fn add(&mut self,p:[f64;3],y:usize){self.n+=1;for k in 0..3{self.correct[k]+=usize::from(usize::from(p[k]>0.5)==y);self.loss[k]-=if y==1{p[k]}else{1.0-p[k]}.clamp(1e-12,1.0).ln();}}fn write(&self,out:&mut String,seed:u64,f:&str,len:usize,phase:usize,measure:&str){for(k,name)in ARMS.iter().enumerate(){out.push_str(&format!("{seed},{f},{len},{phase},{measure},{name},{},{},{:.8},{:.8}\n",self.correct[k],self.n,self.correct[k]as f64/self.n as f64,self.loss[k]/self.n as f64));}}}
#[derive(Clone,Copy,Default)]struct Diag{pre:f64,post:f64,b_mse:f64,n:usize}
fn run(c:&StreamConfig,seed:u64,f:&str,rules:[[f64;2];2],len:usize,out:&mut String,diag:&mut String){
 let contexts:Vec<_>=(0..3*len).map(|t|usize::from(t>=len&&t<2*len)).collect();let data=stream(seed.wrapping_add(40000),&contexts,rules);
 let probe=stream(seed.wrapping_add(48000),&vec![1;c.test_tokens],rules);let mut m=Models::new(seed.wrapping_add(51000),c.learning_rate);let mut slots=[0usize;3];let mut margins=[0.0;3];
 for phase in 0..3{let mut all=Scores::default();let mut first=Scores::default();let mut ds=[[Diag::default();3];2];
  for(i,&(x,y))in data[phase*len..(phase+1)*len].iter().enumerate(){let before=m.state();let p=m.step(x,y,true);let after=m.state();all.add(p,y);if i<100{first.add(p,y);}
   if phase>0{let bucket=usize::from(i>=100);let context=usize::from(phase==1);for k in 0..3{let idx=slots[k]^context;let prior=(1.0-HAZARD)*before[k].0[idx]+HAZARD*before[k].0[1-idx];let post=after[k].0[idx];let b=1-slots[k];let mse=((after[k].1[0][b]-rules[1][0]).powi(2)+(after[k].1[1][b]-rules[1][1]).powi(2))/2.0;let d=&mut ds[bucket][k];d.pre+=1.0-prior;d.post+=1.0-post;d.b_mse+=mse;d.n+=1;}}
  }
  all.write(out,seed,f,len,phase+1,"online");first.write(out,seed,f,len,phase+1,"first_100");
  if phase==0{for(k,(_,rows))in m.state().iter().enumerate(){(slots[k],margins[k])=align_a(*rows,rules[0]);}}
  else{for(bucket,label)in [(0,"first_100"),(1,"remaining")]{for(k,name)in ARMS.iter().enumerate(){let d=ds[bucket][k];diag.push_str(&format!("{seed},{f},{len},{},{label},{name},{},{:.12},{:.12},{:.12},{:.12},{}\n",phase+1,slots[k],margins[k],d.pre/d.n as f64,d.post/d.n as f64,d.b_mse/d.n as f64,d.n));}}}
  if phase==2{let mut frozen=m.clone();frozen.reset();let mut score=Scores::default();for &(x,y)in &probe{score.add(frozen.step(x,y,false),y);}score.write(out,seed,f,len,phase+1,"probe_b");}
 }
}
fn main()->Result<(),Box<dyn std::error::Error>>{let a:Vec<_>=env::args().skip(1).collect();if a.len()!=2{return Err("usage: fixed_slot_diagnostics <config> <new-output-directory>".into());}let c=StreamConfig::parse(&fs::read_to_string(&a[0])?)?;let dir=Path::new(&a[1]);fs::create_dir(dir)?;
 let mut out=String::from("seed,family,phase_length,phase,measurement,model,correct,trials,accuracy,logloss\n");let mut diag=String::from("seed,family,phase_length,phase,window,model,slot_a,alignment_margin,pre_assignment_error,post_assignment_error,b_memory_mse,trials\n");
 for &seed in &c.seeds{for(f,rules)in FAMILIES{for len in LENGTHS{run(&c,seed,f,rules,len,&mut out,&mut diag);}}}fs::write(dir.join("metrics.csv"),out)?;fs::write(dir.join("diagnostics.csv"),diag)?;Ok(())}
#[cfg(test)]mod tests{use super::*;
 #[test]fn mapping_is_selected_once_and_ties_are_deterministic(){assert_eq!(align_a([[0.1,0.9],[0.2,0.8]],[0.1,0.2]).0,0);assert_eq!(align_a([[0.9,0.1],[0.8,0.2]],[0.1,0.2]).0,1);assert_eq!(align_a([[0.5,0.5],[0.5,0.5]],[0.1,0.2]).0,0);}
 #[test]fn predicted_probabilities_do_not_see_target(){let mut a=Models::new(10,0.05);let mut b=a.clone();assert_eq!(a.step(0,0,true),b.step(0,1,true));}
 #[test]fn frozen_probes_leave_learned_rows_unchanged(){let mut a=Models::new(10,0.05);for i in 0..200{a.step(i%2,(i/3)%2,true);}let before=a.state();for i in 0..100{a.step(i%2,(i/7)%2,false);}let after=a.state();for k in 0..3{assert_eq!(before[k].1,after[k].1);}}
}
