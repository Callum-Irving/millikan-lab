use std::error::Error;
use std::path::Path;

pub fn read_one_column(filename: impl AsRef<Path>) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(filename)?;
    let mut points = vec![];

    for record in reader.records() {
        let point = record?[0].parse::<f64>()?;
        points.push(point);
    }

    Ok(points)
}
