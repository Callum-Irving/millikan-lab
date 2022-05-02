mod data;
mod kmeans;

fn main() {
    let filename = "data/charges.csv";
    let max_iters = 15;

    let charges = data::read_one_column(filename).unwrap();

    for k in 1..charges.len() {
        let error = kmeans::kmeans_error(k, &charges, max_iters);
        println!("k = {}, error = {}", k, error * 1E36);
    }

    // 8 seems to be optimal
    let clusters = kmeans::get_clusters(8, &charges, 15);

    // Get average disance between clusters
    let means: Vec<f64> = clusters
        .iter()
        .map(|cluster| {
            // Convert cluster to its mean
            cluster.iter().sum::<f64>() / cluster.len() as f64
        })
        .collect();
    let mut sum = 0.0;
    let len = means.len() as f64 - 1.0;
    for window in means.windows(2) {
        let diff = f64::abs(window[1] - window[0]);
        sum += diff;
    }
    println!("Charge on an electron: {:+e}", sum / len);

    let clusters = clusters
        .into_iter()
        .map(|cluster| {
            cluster
                .into_iter()
                .map(|x| (x, 1.0))
                .collect::<Vec<(f64, f64)>>()
        })
        .collect::<Vec<Vec<(f64, f64)>>>();

    use plotters::prelude::*;
    let root_area = BitMapBackend::new("data/clusters.png", (1280, 720)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Scatter Demo", ("sans-serif", 40))
        .build_cartesian_2d(0.0..2E-18, 0.0..2.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let colours = [&RED, &GREEN, &BLUE];
    for (i, cluster) in clusters.iter().enumerate() {
        let colour = colours[i % colours.len()];
        ctx.draw_series(
            cluster
                .iter()
                .map(|point| Circle::new(*point, 3, Into::<ShapeStyle>::into(colour).filled())),
        )
        .unwrap();
    }
}
