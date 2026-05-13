use icicle_bls12_381::curve::{ScalarCfg as BLS12381ScalarCfg, ScalarField as BLS12381ScalarField};
use icicle_bn254::curve::{ScalarCfg as Bn254ScalarCfg, ScalarField as Bn254ScalarField};
use icicle_runtime::memory::{DeviceVec, HostSlice};

use clap::Parser;
use icicle_core::{
    ntt::{self, initialize_domain},
    traits::{FieldImpl, GenerateRandom},
};
use std::convert::TryInto;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    size: u8,

    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    println!("Running Icicle Examples: Rust NTT");
    let log_size = args.size;
    let size = 1 << log_size;
    println!(
        "---------------------- NTT size 2^{} = {} ------------------------",
        log_size, size
    );

    // Setting Bn254 points and scalars
    println!("Generating random inputs on host for bn254...");
    let scalars = Bn254ScalarCfg::generate_random(size);
    let mut ntt_results = DeviceVec::<Bn254ScalarField>::device_malloc(size).unwrap();

    // Setting bls12381 points and scalars
    println!("Generating random inputs on host for bls12381...");
    let scalars_bls12381 = BLS12381ScalarCfg::generate_random(size);
    let mut ntt_results_bls12381 = DeviceVec::<BLS12381ScalarField>::device_malloc(size).unwrap();

    println!("Setting up bn254 Domain...");
    initialize_domain(
        ntt::get_root_of_unity::<Bn254ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bn254 NTT...");
    let cfg = ntt::NTTConfig::<Bn254ScalarField>::default();

    println!("Setting up bls12381 Domain...");
    initialize_domain(
        ntt::get_root_of_unity::<BLS12381ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bls12381 NTT...");
    let cfg_bls12381 = ntt::NTTConfig::<BLS12381ScalarField>::default();

    println!("Executing bn254 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        HostSlice::from_slice(&scalars),
        ntt::NTTDir::kForward,
        &cfg,
        &mut ntt_results[..],
    )
    .unwrap();
    println!(
        "ICICLE BN254 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    println!("Executing bls12381 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        HostSlice::from_slice(&scalars_bls12381),
        ntt::NTTDir::kForward,
        &cfg_bls12381,
        &mut ntt_results_bls12381[..],
    )
    .unwrap();
    println!(
        "ICICLE BLS12381 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    println!("Moving results to host...");
    let mut host_bn254_results = vec![Bn254ScalarField::zero(); size];
    ntt_results
        .copy_to_host(HostSlice::from_mut_slice(&mut host_bn254_results[..]))
        .unwrap();

    let mut host_bls12381_results = vec![BLS12381ScalarField::zero(); size];
    ntt_results_bls12381
        .copy_to_host(HostSlice::from_mut_slice(&mut host_bls12381_results[..]))
        .unwrap();
}
