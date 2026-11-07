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

// Supervisor allocation 69ce5ca3b44b, recorded before execution. All 300 roles
// are distinct and outside the available 96M--98M prior test ranges; no global
// registry is available. Source revision 40473807b7859079012b571eb8f7af41aae41af4.
// Public similar_rules: A=[.10,.70], B=[.30,.90], 1000 events/phase,
// 1000 frozen B probes, rate .05, 100-event switch windows, bootstrap seed 620013.
const ALLOCATION: [(u64,u64,u64); 100] = [
    (3349449141360630698, 2771776525578133022, 5368059675733346780), // 0
    (4850301673511827396, 7373378388286588202, 7799996444881341811), // 1
    (3559801157024608583, 1303773038984610313, 6227787100863623859), // 2
    (6649547766076787125, 6704646388524067690, 8820840846963269063), // 3
    (4786161560122515682, 4638816564889870325, 7449936124566860278), // 4
    (5093615342243046201, 1340057138040118059, 5514754312125295600), // 5
    (2344016281255928239, 2887471622993371236, 8519946947183612308), // 6
    (3734164109029269521, 4691338511269080194, 1086952632217364048), // 7
    (3028110891512946866, 7476362247472293052, 7396183260198266058), // 8
    (3895795201993635185, 8800948745454353340, 9053038280813615200), // 9
    (2488494555615392326, 6619590621502132570, 8322612995101326404), // 10
    (5399945991885341912, 2101862093647566811, 3374482036418983840), // 11
    (790697491291078334, 3699079885745103304, 6740349814540526159), // 12
    (7702589753056719959, 7693279778736284802, 5898075940213143822), // 13
    (5461561378680665930, 6430448846270711588, 2675062052487398081), // 14
    (293536308414591263, 8126808590143364292, 3005830313683027773), // 15
    (5811383801460726085, 2732720223526142405, 2431146045466475078), // 16
    (8433999706738422077, 6427758022393744696, 4609067181695727073), // 17
    (3936484351250787032, 652601311477196221, 8350623664435711830), // 18
    (6217061051892553984, 8685293953619696327, 4513738920027002389), // 19
    (701475033212235575, 9151153123115859200, 4237804113970786005), // 20
    (6596457284748390089, 6840175600079485668, 4871680911705934562), // 21
    (5998095510730178287, 3176222023696648417, 1195793672592692323), // 22
    (6088997614580383417, 1188244878318579053, 3238424368912081331), // 23
    (3011573360688277066, 2599357628459369643, 3809338030982752449), // 24
    (4269564209954898041, 7188163284599345489, 4455314715100836741), // 25
    (2050948129452485291, 7299840028154292949, 2148969260433285277), // 26
    (9026158003535757235, 2680762226181910139, 6033749834086827051), // 27
    (96614537381374277, 4400695099208047938, 1628007348861164004), // 28
    (4932512155450523661, 5482797015757147753, 5694885132810773508), // 29
    (6611540024341308798, 6474136907214371458, 1104792074316010929), // 30
    (6181173085982600399, 5918105527242812478, 6268473340977980471), // 31
    (3872136573079749429, 5749672465965064861, 5049925429781589891), // 32
    (1991289780836240611, 8585220298115174612, 4193197427724854012), // 33
    (8293236735525633732, 3814051099192628582, 1951884833737431860), // 34
    (2541386762560018688, 4766645893548499962, 6025026692017483985), // 35
    (2628019810652096424, 1381404230938581475, 3968931161788024894), // 36
    (3206128274454029343, 82889517710795352, 7915551225782919840), // 37
    (6212634068117055548, 6499152285130727287, 5126237141214597544), // 38
    (7456123841214586419, 6601966766354217895, 2777814982685609852), // 39
    (1776252683573445022, 5789089959878371665, 1128183028116739642), // 40
    (7928879499359906390, 5776815247877620257, 5698217389676622990), // 41
    (5302103368561078622, 259047856181646395, 7917065337637290275), // 42
    (1006389106846600720, 4573773791337324102, 7061349896769952070), // 43
    (6336489906622848211, 1796983885339968418, 3913179149836507069), // 44
    (932744500471637735, 8016725523422764785, 7366833245484646238), // 45
    (2153339873263782701, 4027489157301596362, 4990300496428818386), // 46
    (2052565328748237192, 5147548384106284434, 8890440616628522179), // 47
    (891475706722356410, 6552446149094824735, 3410075222329837534), // 48
    (6826904658834866514, 3287172916959246588, 2464726462454710230), // 49
    (9080741155708673313, 1457370258891656741, 4798405424163076186), // 50
    (117921547856135978, 6936573482761201948, 6188710993328475029), // 51
    (7049497570671803122, 4365387187880695883, 7409431265228004132), // 52
    (2966549246936460173, 612418913074231383, 6740154990316886992), // 53
    (6845399091219484652, 3044703527012653066, 1877075580008364903), // 54
    (4279100276146748089, 6467988893255382805, 1979448172073305998), // 55
    (1022582800416748565, 4097841502225780429, 4783310730354270891), // 56
    (5090056829870076144, 1791708460380709667, 2239591013841919215), // 57
    (8997405771711021819, 2505979693769460802, 5159840942827314233), // 58
    (5280168454252987594, 3888633586784689491, 6514517596960683706), // 59
    (4340379097437562658, 5547710805628025535, 3926809316031583464), // 60
    (9030710813456216641, 4809235482574646280, 4439123744677630452), // 61
    (1468324217579682216, 3773193331091982152, 7231829495996227282), // 62
    (8762725882966369709, 8314758834312324494, 3793057383172139260), // 63
    (3185183389275802568, 6876519208677465223, 1946038727758247970), // 64
    (3415877910448676446, 2147186677893550621, 7937259132017688238), // 65
    (6723793912346893520, 5584674852139285444, 1097130882307996587), // 66
    (8501875674414865778, 5526694951260925306, 6471509055744805602), // 67
    (7255666183221523912, 7477713181635672188, 7195712425222394198), // 68
    (4563486380986688893, 7754076866593124586, 3451151293903377102), // 69
    (4765688187384818212, 5355766869855003670, 8803953864822964831), // 70
    (6232388889474959162, 2122504187543664804, 7543416559775924428), // 71
    (3788180875991260229, 5003481841052732403, 3218941821831085038), // 72
    (3950859279131052723, 5898264391396846154, 4243100621274093339), // 73
    (5127314813292566604, 7949326281821394305, 9221969302060629323), // 74
    (3231494261597613443, 891209127468129535, 4925150879328029383), // 75
    (882231014879010556, 787528260804747007, 3361389954097901110), // 76
    (1341235829747363998, 2425436569984766202, 151911719901825639), // 77
    (1869598795341427291, 9096489149562062535, 4303728109748258979), // 78
    (5785975636551478904, 2939720482681534802, 8945065892796028810), // 79
    (4268047221391206811, 8961949807097501625, 6291394921705792998), // 80
    (7406245852999140350, 6184204499782117548, 3990283482790601100), // 81
    (8577810945339556647, 8516772053655678192, 881548490371522726), // 82
    (1367726887016075064, 6136290449104710185, 7779200974025476727), // 83
    (1295641229846355941, 3764861302572478242, 1860519636301237484), // 84
    (299872337726245766, 384587605908429370, 3496358099257858617), // 85
    (4653419069443490705, 4654421826622974958, 6443543429870254391), // 86
    (2484440134919006416, 5938270808009421752, 8330613063246608495), // 87
    (3393592086483570156, 3998578376820299969, 4487940317269031348), // 88
    (9023393818871465660, 4394180339123114818, 3539213162695862173), // 89
    (321336153533612028, 8588568305906649272, 4258930851400807706), // 90
    (8089607599415807814, 99541609690092246, 2922101649652860705), // 91
    (4795131916131404394, 4614620657607627931, 1006570500051086002), // 92
    (915033796114896710, 6354340006118572188, 5091793590761900455), // 93
    (2080634923266067362, 642765101326420095, 8742641702853287979), // 94
    (1343333909456450034, 6597858401549505754, 7997885165406120744), // 95
    (3173298663196980223, 7849225322781383768, 7803007892264687120), // 96
    (7796862566114275086, 9178020466556777320, 8893987246236110251), // 97
    (4186763297129488695, 342461432033332720, 339720662975034266), // 98
    (5015031256050245721, 1786297559295639531, 8152838630566278615), // 99
];

#[test]
fn similar_rules_allocated_filter_ablation() {
    const N: usize = 100;
    const LENGTH: usize = 1000;
    let rules = [[0.10, 0.70], [0.30, 0.90]];
    let mut counts = [[[0usize; 3]; 3]; N];
    for (i, &(stream, init, probe)) in ALLOCATION.iter().enumerate() {
        let mut filtered = Candidate::new(init, 0.05);
        let mut unguarded = filtered.unguarded_smoothing();
        let mut mixture = Mixture::new(init, 0.05);
        let mut rng = Rng(stream);
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let y = usize::from(rng.uniform() < rules[usize::from(phase == 1)][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, true).0];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..3 { counts[i][phase-1][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            }
            filtered.observe(x, y, true); unguarded.observe(x, y, true); x = y;
        }
        filtered.reset(); unguarded.reset(); mixture.reset();
        let mut rng = Rng(probe);
        let mut x = (rng.next_u64() & 1) as usize;
        for _ in 0..LENGTH {
            let y = usize::from(rng.uniform() < rules[1][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, false).0];
            for k in 0..3 { counts[i][2][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            filtered.observe(x, y, false); unguarded.observe(x, y, false); x = y;
        }
        println!("mapping seed={i} stream={stream} init={init} probe={probe} switch_b={:?}/100 switch_a={:?}/100 recall_b={:?}/1000", counts[i][0], counts[i][1], counts[i][2]);
        println!("MURMUR_PAIRED {{\"measure\":\"similar_rules_retention_b\",\"candidate\":\"consistency_filter\",\"control\":\"unguarded\",\"seed\":{i},\"candidate_correct\":{},\"control_correct\":{},\"trials\":1000}}", counts[i][2][0], counts[i][2][1]);
    }
    for (w, name) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let trials = if w == 2 { 1000 } else { 100 };
        for (candidate, control, a, b) in [("consistency_filter", "unguarded", 0, 1), ("consistency_filter", "mixture", 0, 2), ("unguarded", "mixture", 1, 2)] {
            let ca: usize = counts.iter().map(|c| c[w][a]).sum();
            let cb: usize = counts.iter().map(|c| c[w][b]).sum();
            let diffs: Vec<f64> = counts.iter().map(|c| (c[w][a] as f64-c[w][b] as f64)/trials as f64).collect();
            let mean = diffs.iter().sum::<f64>() / N as f64;
            let mut rng = Rng(620013 + (w * 9 + a * 3 + b) as u64);
            let mut boot = Vec::with_capacity(10_000);
            for _ in 0..10_000 { boot.push((0..N).map(|_| diffs[(rng.next_u64() % N as u64) as usize]).sum::<f64>() / N as f64); }
            boot.sort_by(f64::total_cmp);
            let (lo, hi) = (boot[250], boot[9749]);
            println!("paired {name} {candidate} minus {control}: mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}] bootstrap_seed={}", 620013 + (w * 9 + a * 3 + b) as u64);
            if w == 2 && a == 0 && b == 1 { println!("primary decision: {}", if lo > 0.0 { "recall benefit" } else if hi < 0.0 { "recall loss" } else { "inconclusive" }); }
            println!("MURMUR_FOLLOWUP {{\"measure\":\"similar_rules_{name}\",\"candidate\":\"{candidate}\",\"control\":\"{control}\",\"candidate_correct\":{ca},\"control_correct\":{cb},\"trials\":{}}}", N*trials);
        }
    }
}
