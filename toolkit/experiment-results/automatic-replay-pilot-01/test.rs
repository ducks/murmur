use murmur::data::Rng;
use murmur_research_runner::{agent_candidate::Candidate, unlabeled_context::Mixture};

#[test]
fn switch_guard_and_frozen_learning() {
    let p = [0.99, 0.01];
    let stable = Candidate::persistence_probability(p, [0.9, 0.1], 0.05);
    let changed = Candidate::persistence_probability(p, [0.01, 0.99], 0.05);
    assert!(stable > changed && changed > 0.0 && stable < 1.0);
    let mut model = Candidate::new(71, 0.05).unguarded_smoothing().guarded_smoothing();
    let mut control = model.unguarded_smoothing();
    for i in 0..300 {
        let x = i % 2;
        let y = if i % 9 < 6 { x } else { 1-x };
        model.observe(x, y, true);
        control.observe(x, y, true);
    }
    model.reset(); control.reset();
    assert!((0..2).any(|x| (model.predict(x)-control.predict(x)).abs() > 1e-10));
    let mut frozen = model.clone();
    for i in 0..100 { frozen.observe(i % 2, (i / 3) % 2, false); }
    frozen.reset(); model.reset();
    for x in 0..2 { assert_eq!(frozen.predict(x), model.predict(x)); }
}

#[test]
fn consistency_filter_changes_learning_but_not_frozen_parameters() {
    let mut filtered = Candidate::new(871, 0.05);
    let mut unguarded = filtered.unguarded_smoothing();
    for i in 0..600 {
        let x = (i / 3) % 2;
        let y = usize::from(i % 11 < 7) ^ x;
        filtered.observe(x, y, true);
        unguarded.observe(x, y, true);
    }
    filtered.reset(); unguarded.reset();
    assert!((0..2).any(|x| (filtered.predict(x)-unguarded.predict(x)).abs() > 1e-10));
    let mut frozen = filtered.clone();
    for i in 0..100 { frozen.observe(i % 2, (i / 3) % 2, false); }
    frozen.reset();
    for x in 0..2 { assert_eq!(frozen.predict(x), filtered.predict(x)); }
}

// Prespecified seed ranges (inclusive): streams 97_100_000..=97_100_999,
// initialization 97_200_000..=97_200_999, probes 97_300_000..=97_300_999.
// These do not overlap the preceding 96-million-range replication. Each seed is a resampling unit.
#[test]
fn high_noise_consistency_filter_replication() {
    const N: usize = 1000;
    const LENGTH: usize = 200;
    const PROBE: usize = 1000;
    const BOOTSTRAPS: usize = 10_000;
    let rules = [[0.3, 0.7], [0.7, 0.3]];
    // Model indices: filter, unguarded, fixed mixture. Outcomes: B, A, recall.
    let mut totals = [[0usize; 3]; 3];
    let mut per_seed = [[[0usize; 3]; 3]; N];
    for seed in 0..N {
        let mut rng = Rng(97_100_000 + seed as u64);
        let mut filtered = Candidate::new(97_200_000 + seed as u64, 0.05);
        let mut unguarded = filtered.unguarded_smoothing();
        let mut mixture = Mixture::new(97_200_000 + seed as u64, 0.05);
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let y = usize::from(rng.uniform() < rules[usize::from(phase == 1)][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, true).0];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..3 {
                    per_seed[seed][phase - 1][k] += usize::from(usize::from(predictions[k] > 0.5) == y);
                }
            }
            filtered.observe(x, y, true);
            unguarded.observe(x, y, true);
            x = y;
        }
        filtered.reset(); unguarded.reset(); mixture.reset();
        let mut probe = Rng(97_300_000 + seed as u64);
        let mut x = (probe.next_u64() & 1) as usize;
        for _ in 0..PROBE {
            let y = usize::from(probe.uniform() < rules[1][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, false).0];
            for k in 0..3 {
                per_seed[seed][2][k] += usize::from(usize::from(predictions[k] > 0.5) == y);
            }
            filtered.observe(x, y, false);
            unguarded.observe(x, y, false);
            x = y;
        }
        println!("seed={seed} stream={} init={} probe={} switch_b={:?} switch_a={:?} recall_b={:?}", 97_100_000 + seed, 97_200_000 + seed, 97_300_000 + seed, per_seed[seed][0], per_seed[seed][1], per_seed[seed][2]);
        for w in 0..3 { for k in 0..3 { totals[w][k] += per_seed[seed][w][k]; } }
    }
    for (w, label) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let per_trial = if w == 2 { PROBE } else { 100 };
        let trials = N * per_trial;
        for (candidate, control, a, b) in [("consistency_filter", "unguarded", 0, 1),
                                           ("consistency_filter", "mixture", 0, 2),
                                           ("unguarded", "mixture", 1, 2)] {
            let differences: Vec<f64> = per_seed.iter().map(|c|
                (c[w][a] as f64 - c[w][b] as f64) / per_trial as f64).collect();
            let mean = differences.iter().sum::<f64>() / N as f64;
            let mut rng = Rng(97_400_000 + (w * 9 + a * 3 + b) as u64);
            let mut bootstrap = Vec::with_capacity(BOOTSTRAPS);
            for _ in 0..BOOTSTRAPS {
                let sum: f64 = (0..N).map(|_| differences[(rng.next_u64() % N as u64) as usize]).sum();
                bootstrap.push(sum / N as f64);
            }
            bootstrap.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (bootstrap[250], bootstrap[9749]);
            println!("paired {label} {candidate}-minus-{control}: mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}]");
            if w == 2 && a == 0 && b == 1 {
                let decision = if mean >= 0.01 && lo > 0.0 { "supports benefit" }
                    else if mean <= -0.01 && hi < 0.0 { "weighs against filter" }
                    else { "inconclusive" };
                println!("primary decision: {decision}");
            }
            if b == 2 { println!("mixture criterion {label} {candidate}: {}", if mean < -0.02 { "flag" } else { "pass" }); }
            println!("MURMUR_FOLLOWUP {{\"measure\":\"high_noise_200_{label}\",\"candidate\":\"{candidate}\",\"control\":\"{control}\",\"candidate_correct\":{},\"control_correct\":{},\"trials\":{trials}}}", totals[w][a], totals[w][b]);
        }
    }
}

// Allocation 44758bcf7c0f, recorded by supervisor before execution. Roles
// checked against the prior allocation and documented 96M--98M ranges; no complete
// historical registry exists. Source revision 653e854a969357c0a0a595d292d6ef0fc1c5882b.
// similar_rules: A=[.10,.70], B=[.30,.90], 1000 events/phase,
// 1000 frozen B probes, rate .05, switch windows 100, bootstrap seed 720019.
const FOLLOWUP_ALLOCATION: [(u64,u64,u64); 100] = [
    (8438911934030753459, 6160750815251356880, 75077856414359430), // 0
    (5607261566012881703, 6008914987270902930, 6084178198333762944), // 1
    (6588479744907302274, 6725304490281087901, 8594626650150772584), // 2
    (5518599966878049001, 1343764812210695147, 8607785954189797379), // 3
    (6600863914135308505, 1477624178118932451, 264753963450382786), // 4
    (8315856259400061125, 5920335904027408825, 7591248873018532147), // 5
    (2334880850832762633, 9111956424721733548, 4995163902251784222), // 6
    (8372362895553650485, 5109045955783282923, 2719764552533123585), // 7
    (487794457753298436, 9076008420230613366, 7886927220298303926), // 8
    (6634092659091940001, 8031353987846815010, 3736592181326244101), // 9
    (5559958290963716078, 1395086017560856219, 2692151938369718815), // 10
    (1778838533397454435, 511517751381681243, 8315827655137542102), // 11
    (5155310966602595289, 4333340115462514278, 4806776010706267660), // 12
    (3145216947491999893, 1966744034694131430, 3517187464407775985), // 13
    (9057131150462724855, 8791221848257596246, 7918268045911535912), // 14
    (1545633175148782283, 8148023090163623777, 3356142983533040708), // 15
    (905838616541529754, 8155535417295619678, 3218534951317752618), // 16
    (3094174195281853161, 6305522658570666543, 1106508058011345552), // 17
    (1910471091241899469, 1030619739247649742, 9014177476947569565), // 18
    (4542907737963945701, 2077954324181597767, 4815073179358798140), // 19
    (2008890128871635679, 4830487290793305094, 7231714568297508805), // 20
    (2697151389527891184, 1703096465162292717, 7390980744566122089), // 21
    (3537120776128204688, 4797441709238923881, 8586297696678398957), // 22
    (4704104615574987768, 6219319477501433932, 6233856372046969664), // 23
    (5582805035014073148, 1777431456588086488, 4169731584506668092), // 24
    (2334473192367472173, 4081175609163928140, 4785907509169748296), // 25
    (8805894575498906901, 3739005993376811146, 3070846165071224337), // 26
    (7795314969661226458, 1027167410740796803, 5434925731141581783), // 27
    (7310594554333436774, 6517305512284371183, 9055690209014514448), // 28
    (7272432535655017965, 5238037527410782065, 6683377108313780496), // 29
    (829212430002862161, 8488358852565521719, 4638600635407479790), // 30
    (4612240224446581057, 62259155269660028, 2609087051096587005), // 31
    (827531050104041734, 5561072965696288775, 5824064306144961128), // 32
    (4045671281983919, 6525294057713314409, 126073910665521844), // 33
    (6935621407666958218, 5787084273051056925, 3556496550087246471), // 34
    (7455586274110912287, 7857667412898741901, 3597979196965045852), // 35
    (1592108725621267580, 6434996455287939456, 7709113025884444322), // 36
    (6844980128406223993, 3019825580260393482, 4711806753203443956), // 37
    (478431110440610652, 3233239796987554630, 1707829350258427349), // 38
    (7635974886279657909, 2770728881607371107, 7289286776852655594), // 39
    (4798794415240634545, 4992286151215783337, 1515104260960586907), // 40
    (8759961719669930967, 1330851403538523989, 1387373751272999317), // 41
    (1555494820815076886, 5614162970528927949, 4690596736976300108), // 42
    (3935783086168979586, 656455139747685153, 7695642931801246030), // 43
    (7127134714207508815, 1826419162895861206, 1373829323767468164), // 44
    (8139814789268114786, 3381204414869181566, 1583798428416378299), // 45
    (6869339744088589622, 5193921728029392184, 8084257786690135646), // 46
    (1192968489983501378, 843104859224061576, 413875657791821095), // 47
    (1259172847006303836, 7696979265847951145, 868253532243455775), // 48
    (5686566671265218041, 426407657132487214, 2189344510423709345), // 49
    (4288193861571775394, 6901311547964117285, 776937070282304864), // 50
    (1416095272098999956, 7265654627608943233, 5256081358766514615), // 51
    (5193835773773756056, 8135860929670378936, 2229952758739725828), // 52
    (3551092125397957871, 5251917718706925968, 3445879972071688352), // 53
    (950907843953100000, 8249389538755646203, 8267770214147963809), // 54
    (5599628092202190283, 123136632993206649, 7073285502750886346), // 55
    (6226306112285740942, 6976245157828623656, 2444735883993839546), // 56
    (3066435258798606264, 4304871344028441587, 7171498821026728380), // 57
    (1503914862169430144, 8827156886194542662, 2933352279256982438), // 58
    (985970095176009991, 1081597668960785544, 804780836127626362), // 59
    (5396908565668462608, 6587385488614329541, 4232485761420444907), // 60
    (8700099924530917789, 8585197083626118981, 1869735980069595745), // 61
    (3754036219771063159, 2630383715705714962, 3793393250724316990), // 62
    (7788153725104225292, 6580774618641851126, 5768522686257242440), // 63
    (98125977004326190, 7606875202793256155, 1193583727737824432), // 64
    (4320541805125798956, 6692695773526025669, 1942411098313639288), // 65
    (1936150680150710413, 5671688110272123265, 1929499415236230411), // 66
    (8509889041644820271, 1680166629401068900, 5980321208574718864), // 67
    (811139654026605353, 6895932233297120679, 5837149772193634457), // 68
    (7720919957247616523, 430942823252524333, 7678352457840306940), // 69
    (3557766729841054375, 5450152577658037799, 6823141735988879378), // 70
    (7243278703368647126, 7030113880200345577, 5024521228256216595), // 71
    (5853402403137229763, 4507316384577310046, 685587598574161893), // 72
    (5726245257814281533, 7034885218448480608, 9213326351353927764), // 73
    (107121594433964190, 4189541413969782317, 1215940694635842989), // 74
    (534525295259489621, 472246790598842352, 2011198292371082581), // 75
    (7561798750393294784, 1852728816697211304, 4768396421794289176), // 76
    (5151284352731608973, 2470774047447684118, 6276882222669784818), // 77
    (5156669888132651543, 2729130619586990467, 2541230704772754556), // 78
    (7631307004740995700, 1108699695504442274, 4892447890780423681), // 79
    (1211331358468708489, 9006960833478185524, 9195544533389665641), // 80
    (1087032761439711086, 169756363652877855, 4737958312017075834), // 81
    (5946031334583144573, 1224744513244393290, 1533664312781533325), // 82
    (6240286733865782399, 3976380765847835434, 7416054756686889303), // 83
    (4076677809327521582, 1720969261055018635, 2008364498500509544), // 84
    (3708093917396489349, 4065723891148773212, 915979272486619695), // 85
    (5645610727867133110, 3733156515997432490, 6292324318148573068), // 86
    (7980841591754079206, 3846596229315210076, 4018946096601605634), // 87
    (1905032645043250521, 4658341265319327038, 3335510431645513260), // 88
    (4821197502986375368, 2241224798657010678, 1489985228709637217), // 89
    (3748778372647423048, 5091074448388425485, 7413689797173502338), // 90
    (4346829477285956500, 7438541859225722849, 778460104288043684), // 91
    (6680867224460152515, 1306758492961753147, 3350807929315202428), // 92
    (8650802049496566912, 165564311889761710, 1050813010580233003), // 93
    (2253760369658528399, 6335822298983288585, 5909573042749055226), // 94
    (3927894502882440170, 5839983122960599307, 6144152964935239256), // 95
    (4987609123558216777, 2678239842119947573, 2339677730582182220), // 96
    (3798093222867232205, 3246913594042593866, 8882228133909561482), // 97
    (7690031126082444332, 9049517688221300197, 6477969849771878038), // 98
    (7474910814036769587, 3883128961930912183, 5419177062888181233), // 99
];

#[test]
fn similar_rules_unguarded_mixture_allocated_followup() {
    const N: usize = 100;
    const LENGTH: usize = 1000;
    const BOOT_SEED: u64 = 720019;
    let rules = [[0.10, 0.70], [0.30, 0.90]];
    let mut counts = [[[0usize; 2]; 3]; N];
    for (i, &(stream, init, probe)) in FOLLOWUP_ALLOCATION.iter().enumerate() {
        let mut candidate = Candidate::new(init, 0.05).unguarded_smoothing();
        let mut mixture = Mixture::new(init, 0.05);
        let mut rng = Rng(stream);
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let y = usize::from(rng.uniform() < rules[usize::from(phase == 1)][x]);
            let predictions = [candidate.predict(x), mixture.step(x, y, true).0];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..2 { counts[i][phase - 1][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            }
            candidate.observe(x, y, true);
            x = y;
        }
        candidate.reset(); mixture.reset();
        let mut rng = Rng(probe);
        let mut x = (rng.next_u64() & 1) as usize;
        for _ in 0..LENGTH {
            let y = usize::from(rng.uniform() < rules[1][x]);
            let predictions = [candidate.predict(x), mixture.step(x, y, false).0];
            for k in 0..2 { counts[i][2][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            candidate.observe(x, y, false);
            x = y;
        }
        println!("mapping seed={i} stream={stream} init={init} probe={probe} switch_b={:?}/100 switch_a={:?}/100 recall_b={:?}/1000", counts[i][0], counts[i][1], counts[i][2]);
        println!("MURMUR_PAIRED {{\"measure\":\"similar_rules_switch_to_b\",\"candidate\":\"unguarded\",\"control\":\"mixture\",\"seed\":{i},\"candidate_correct\":{},\"control_correct\":{},\"trials\":100}}", counts[i][0][0], counts[i][0][1]);
    }
    for (w, name) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let trials = if w == 2 { LENGTH } else { 100 };
        let ca: usize = counts.iter().map(|c| c[w][0]).sum();
        let cb: usize = counts.iter().map(|c| c[w][1]).sum();
        let diffs: Vec<f64> = counts.iter().map(|c| (c[w][0] as f64 - c[w][1] as f64) / trials as f64).collect();
        let mean = diffs.iter().sum::<f64>() / N as f64;
        let mut rng = Rng(BOOT_SEED + w as u64);
        let mut boot = Vec::with_capacity(10_000);
        for _ in 0..10_000 { boot.push((0..N).map(|_| diffs[(rng.next_u64() % N as u64) as usize]).sum::<f64>() / N as f64); }
        boot.sort_by(f64::total_cmp);
        let (lo, hi) = (boot[250], boot[9749]);
        println!("paired {name} unguarded minus mixture: mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}] draws=10000 bootstrap_seed={}", BOOT_SEED + w as u64);
        if w == 0 { println!("primary decision: {}", if lo > -0.02 { "non-inferiority supported" } else if hi < -0.02 { "loss beyond margin" } else { "inconclusive" }); }
        println!("MURMUR_FOLLOWUP {{\"measure\":\"similar_rules_{name}\",\"candidate\":\"unguarded\",\"control\":\"mixture\",\"candidate_correct\":{ca},\"control_correct\":{cb},\"trials\":{}}}", N * trials);
    }
}
