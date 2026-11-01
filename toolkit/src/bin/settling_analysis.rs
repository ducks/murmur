//! Post-hoc descriptive analysis of predictive_graph's own convergence flag.
//! This reader expects the runner's unquoted numeric CSV, not arbitrary CSV.
use std::{collections::BTreeMap, env, fs};
#[derive(Default, Clone, Copy)]
struct Group {
    n: usize,
    graph: usize,
    control: usize,
}
fn summarize(input: &str) -> Result<String, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or("empty input")?;
    let cols: Vec<_> = header.split(',').collect();
    let find = |name: &str| {
        cols.iter()
            .position(|v| *v == name)
            .ok_or_else(|| format!("missing column: {name}"))
    };
    let (seed, target, graph, control, settled) = (
        find("seed")?,
        find("target")?,
        find("predictive")?,
        find("predictive_no_feedback")?,
        find("predictive_converged")?,
    );
    let mut groups: BTreeMap<(u64, bool), Group> = BTreeMap::new();
    let mut pooled = [Group::default(); 2];
    let mut count = 0;
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<_> = line.split(',').collect();
        if row.len() != cols.len() {
            return Err(format!("line {}: wrong field count", i + 2));
        }
        let parse = |col: usize| {
            row[col]
                .parse::<u64>()
                .map_err(|_| format!("line {}: invalid integer", i + 2))
        };
        let id = parse(seed)?;
        let target = parse(target)?;
        let g = parse(graph)?;
        let c = parse(control)?;
        let flag = match row[settled] {
            "true" => true,
            "false" => false,
            _ => return Err(format!("line {}: invalid convergence flag", i + 2)),
        };
        if target > 3 || g > 3 || c > 3 {
            return Err(format!("line {}: symbol outside 0..3", i + 2));
        }
        for x in [
            groups.entry((id, flag)).or_default(),
            &mut pooled[usize::from(flag)],
        ] {
            x.n += 1;
            x.graph += usize::from(g == target);
            x.control += usize::from(c == target);
        }
        count += 1;
    }
    if count == 0 {
        return Err("no prediction rows".into());
    }
    let mut out=String::from("scope,seed,settled,trials,graph_correct,no_feedback_correct,graph_accuracy,no_feedback_accuracy\n");
    let format_row = |scope: &str, id: &str, flag: bool, x: Group| {
        format!(
            "{scope},{id},{flag},{},{},{},{},{}\n",
            x.n,
            x.graph,
            x.control,
            if x.n == 0 {
                String::new()
            } else {
                (x.graph as f64 / x.n as f64).to_string()
            },
            if x.n == 0 {
                String::new()
            } else {
                (x.control as f64 / x.n as f64).to_string()
            }
        )
    };
    for ((id, flag), x) in groups {
        out.push_str(&format_row("seed", &id.to_string(), flag, x));
    }
    for flag in [false, true] {
        out.push_str(&format_row("pooled", "", flag, pooled[usize::from(flag)]));
    }
    Ok(out)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: settling_analysis <predictive-predictions.csv>".into());
    }
    print!("{}", summarize(&fs::read_to_string(&args[0])?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::summarize;
    #[test]
    fn counts_are_stratified_and_bad_data_is_rejected() {
        let data="seed,target,predictive,predictive_no_feedback,predictive_converged\n1,1,1,0,true\n1,2,0,2,false\n";
        let out = summarize(data).unwrap();
        assert!(out.contains("pooled,,true,1,1,0,1,0"));
        assert!(out.contains("pooled,,false,1,0,1,0,1"));
        assert!(summarize("seed,target\n1,0\n").is_err());
        assert!(summarize(&data.replace("true", "maybe")).is_err());
    }
}
