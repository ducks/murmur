//! Bayesian online change-point detection with transition-count emissions.
//! A fixed-hazard BOCPD reference, compared against a simple CUSUM-like alarm.
//! Neither detector creates a memory. Thresholds must be calibrated empirically.
use std::{env,fs};
#[derive(Clone)] struct Rng(u64);
impl Rng {
 fn next(&mut self)->u64 { self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31) }
 fn uniform(&mut self)->f64 {(self.next()>>11) as f64/((1u64<<53) as f64)}
}
#[derive(Clone,Copy)] struct Event { source:usize,target:usize }
fn stream(seed:u64,n:usize,block:Option<usize>,noise:f64)->Vec<Event> {
 let mut rng=Rng(seed);let mut source=(rng.next()%4) as usize;let mut out=Vec::with_capacity(n);
 for i in 0..n {let context=block.map_or(0,|b|(i/b)%2);let desired=if context==0 {(source+1)%4}else{(source+3)%4};let target=if rng.uniform()<noise {let a=(rng.next()%3) as usize;if a>=desired {a+1}else{a}}else{desired};out.push(Event{source,target});source=target;}out
}
const HAZARD:f64=0.01; const MAX_RUN:usize=256;
#[derive(Clone)] struct Hyp { mass:f64, counts:[[u32;4];4] }
struct Bocpd { hypotheses:Vec<Hyp>, last_cp:f64 }
impl Bocpd {
 fn new()->Self {Self{hypotheses:vec![Hyp{mass:1.0,counts:[[0;4];4]}],last_cp:0.0}}
 fn step(&mut self,e:Event)->f64 {
  // A change occurs before this transition: new segment predictive is uniform.
  let cp=HAZARD*0.25;
  let mut growth=Vec::with_capacity(self.hypotheses.len()+1);
  let mut fresh=Hyp{mass:cp,counts:[[0;4];4]};fresh.counts[e.source][e.target]+=1;growth.push(fresh);
  for h in &self.hypotheses {
   let row=&h.counts[e.source];let sum:u32=row.iter().sum();let p=(row[e.target] as f64+0.5)/(sum as f64+2.0);
   let mut next=h.clone();next.mass=h.mass*(1.0-HAZARD)*p;next.counts[e.source][e.target]+=1;growth.push(next);
  }
  // Truncate tiny tails and renormalize; no exact posterior guarantee after cap.
  if growth.len()>MAX_RUN {growth.pop();}
  let total:f64=growth.iter().map(|h|h.mass).sum();for h in &mut growth {h.mass/=total;}
  self.last_cp=growth[0].mass;self.hypotheses=growth;self.last_cp
 }
}
struct Cusum { counts:[[u64;4];4], evidence:f64, steps:usize }
impl Cusum {fn new()->Self {Self{counts:[[1;4];4],evidence:0.0,steps:0}} fn step(&mut self,e:Event)->f64 {
 let row=&self.counts[e.source];let p=row[e.target] as f64 /row.iter().sum::<u64>() as f64;self.steps+=1;
 if self.steps>200 {self.evidence=(self.evidence+(0.25/p).ln()).max(0.0);}self.counts[e.source][e.target]+=1;self.evidence
}}
const B_THRESH:[f64;6]=[0.02,0.04,0.08,0.16,0.32,0.64];
const C_THRESH:[f64;6]=[5.0,10.0,20.0,40.0,80.0,160.0];
fn thresholds(b:&[f64],c:&[f64],block:Option<usize>)->Vec<(usize,f64,bool,usize,usize)> {
 let mut out=Vec::new();
 for (kind,values,thresh) in [(0,b,&B_THRESH),(1,c,&C_THRESH)] {
  for &t in thresh {
   let mut prev=false;let mut alarms=Vec::new();
   for (i,&x) in values.iter().enumerate(){let on=i>=200 && x>t;if on&&!prev {alarms.push(i+1)} prev=on;}
   let mut hits=0;let mut switches=0;
   if let Some(k)=block {for start in (k..values.len()).step_by(k) {let step=start+1;if step<=200 {continue;} switches+=1;if alarms.iter().any(|a|*a>=step && *a<step+20){hits+=1;}}}
   out.push((kind,t,!alarms.is_empty(),hits,switches));
  }
 } out
}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=env::args().skip(1).collect();if args.len()!=2{return Err("usage: change_detection SEED OUTPUT.csv".into());}let seed:u64=args[0].parse()?;
 let mut csv=String::from("seed,case,noise,block,length,detector,threshold,any_alarm,switches_detected,switches_total\n");
 for (name,block,noise,n) in [("stable10",None,0.1,5000),("stable25",None,0.25,5000),("switch50",Some(50),0.1,1200),("switch100",Some(100),0.1,1200),("switch200",Some(200),0.1,1200)] {
  let mut model=Bocpd::new();let mut baseline=Cusum::new();let mut b=Vec::new();let mut c=Vec::new();
  for e in stream(seed.wrapping_add(11),n,block,noise) {b.push(model.step(e));c.push(baseline.step(e));}
  for (kind,threshold,alarm,hits,total) in thresholds(&b,&c,block) {csv.push_str(&format!("{seed},{name},{noise},{},{n},{},{threshold},{alarm},{hits},{total}\n",block.unwrap_or(0),if kind==0 {"bocpd"}else{"cusum"}));}
 }
 fs::write(&args[1],&csv)?;print!("{csv}");Ok(())
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn posterior_normalized_and_bounded() {let mut b=Bocpd::new();for e in stream(7,1000,Some(100),0.1){let p=b.step(e);assert!((0.0..=1.0).contains(&p));assert!((b.hypotheses.iter().map(|h|h.mass).sum::<f64>()-1.0).abs()<1e-10);assert!(b.hypotheses.len()<=MAX_RUN);}}
 #[test] fn seeded_stream_repeatable(){let a=stream(17,500,Some(50),0.25);let b=stream(17,500,Some(50),0.25);assert!(a.iter().zip(b).all(|(x,y)| x.source==y.source&&x.target==y.target));}
}
