extern crate yaml_rust;
use yaml_rust::{YamlLoader};

extern crate multiset;
use multiset::HashMultiSet;

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {

    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn create_spec(opname: &str, meta: &str) -> String {
    format!("{{opname: \"{}\", meta: \"{}\"}}", opname, meta)
}

fn main() {
    let filename = std::env::args().nth(1).expect("No filename provided!");
    let varname = std::env::args().nth(2).expect("No filename provided!");

    let mut include_set = HashSet::new();
    include_set.insert("_FusedConv2D");
    include_set.insert("Conv2D");
    include_set.insert("Conv2DBackpropInput");
    include_set.insert("Conv2DBackpropFilter");
    include_set.insert("MatMul");
    include_set.insert("BatchMatMul");
    include_set.insert("BatchMatMulV2");
    include_set.insert("_FusedMatMul");
    include_set.insert("Tanh");
    include_set.insert("TanhGrad");
    include_set.insert("Sigmoid");
    include_set.insert("SigmoidGrad");
    include_set.insert("Relu");
    include_set.insert("ReluGrad");


    let mut layers: HashMultiSet<String> = HashMultiSet::new();

    let mut total_fp32 : i64 = 0;
    let mut counted_fp32 : i64 = 0;

    let lines = read_lines(filename).unwrap();

    for line in lines {
        if let Ok(line_ok) = line {
            if line_ok.contains("Barrier") {
                layers = HashMultiSet::new();
                total_fp32 = 0;
                counted_fp32 = 0;
                continue;
            }

            let raw_data = YamlLoader::load_from_str(&line_ok).unwrap();
            let yd = &raw_data[0][0];
            if let Some(opname) = yd["opname"].as_str() {
                if yd["rid"].as_i64() == Some(0) {
                    total_fp32 += yd["fp32"].as_i64().unwrap();
                    if include_set.contains(opname) {
                        counted_fp32 += yd["fp32"].as_i64().unwrap();
                        let spec = create_spec(
                            opname, yd["meta"].as_str().unwrap());
                        layers.insert(spec);
                    }
                }
            }
        }
    }

    println!("{}:", varname);
    println!("  total-flops:   {}", total_fp32);
    println!("  counted-flops: {}", counted_fp32);
    println!("  accounted: {}", counted_fp32 as f64 / total_fp32 as f64);

    println!("  layers:");
    for key in layers.distinct_elements() {
        println!("    - {{count: {}, spec: {}}}", layers.count_of(key), key);
    }
    println!("");


}
