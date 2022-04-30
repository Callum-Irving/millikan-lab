/// Compute the SSE of k-means on a 1-dimensional dataset.
pub fn kmeans_error(k: usize, data: &[f32], max_iters: usize) -> f32 {
    // Create initial centroids
    let mut centroids = initialize_centroids(k, data);

    let mut error = 0.0;
    let mut clusters: Vec<Vec<f32>> = vec![vec![]; k];
    for _ in 0..max_iters {
        error = 0.0;

        // Assign each point to a centroid
        for point in data {
            let mut current_centroid = 0;
            let mut best_dist = f32::MAX;
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = f32::abs(centroid - point);
                if distance < best_dist {
                    current_centroid = i;
                    best_dist = distance;
                }
            }
            clusters[current_centroid].push(*point);
        }

        // Move each centroid closer to the center of the cluster
        for (centroid, cluster) in centroids.iter_mut().zip(clusters.iter()) {
            let mean = cluster.iter().sum::<f32>() / cluster.len() as f32;
            *centroid = mean;

            // Compute the sum of squared error
            error += cluster
                .iter()
                .map(|x| f32::abs(x - *centroid).powi(2))
                .sum::<f32>();
        }

        // Reset clusters
        for cluster in clusters.iter_mut() {
            cluster.clear();
        }
    }

    error
}

pub fn get_clusters(k: usize, data: &[f32], max_iters: usize) -> Vec<Vec<f32>> {
    // Create initial centroids
    let mut centroids = initialize_centroids(k, data);

    let mut clusters = vec![vec![]; k];
    for _ in 0..max_iters {
        // Reset clusters
        for cluster in clusters.iter_mut() {
            cluster.clear();
        }

        // Assign each point to a centroid
        for point in data {
            let mut current_centroid = 0;
            let mut best_dist = f32::MAX;
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = f32::abs(centroid - point);
                if distance < best_dist {
                    current_centroid = i;
                    best_dist = distance;
                }
            }
            clusters[current_centroid].push(*point);
        }

        // Move centroids
        for (centroid, cluster) in centroids.iter_mut().zip(clusters.iter()) {
            let mean = cluster.iter().sum::<f32>() / cluster.len() as f32;
            *centroid = mean;
        }
    }

    clusters
}

fn initialize_centroids(k: usize, data: &[f32]) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k);

    let min = data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Evenly distribute over [min, max]

    for i in 0..k {
        // let index = (i as f32 * data.len() as f32 / k as f32) as usize;
        // centroids.push(data[index]);
        centroids.push((max - min) / (k as f32 - 1.0) * i as f32 + min);
    }
    centroids
}
