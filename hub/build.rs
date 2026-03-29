use tonic_prost_build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("../proto/naila.proto")?;
    Ok(())
}
