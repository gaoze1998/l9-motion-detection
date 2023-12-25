use opencv::{prelude::*, videoio::{VideoCapture, self}, highgui, imgproc, core, types::VectorOfVectorOfPoint, imgcodecs::imwrite};

fn motion_detection(frames_in_two_secs: &Vec<Mat>) -> bool {
    let mut gray_frames = vec![Mat::default(); 5];
    for i in 0..2 {
        imgproc::cvt_color(&frames_in_two_secs[i], &mut gray_frames[i], imgproc::COLOR_BGR2GRAY, 0).unwrap();
    }
    let mut diff = Mat::default();
    core::absdiff(&gray_frames[0], &gray_frames[1], &mut diff).unwrap();
    let mut thresh = Mat::default();
    imgproc::threshold(&diff, &mut thresh, 25.0, 255.0, imgproc::THRESH_BINARY).unwrap();
    let mut dilated = Mat::default();
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(3, 3), core::Point::new(-1, -1)).unwrap();
    imgproc::dilate(&thresh, &mut dilated, &kernel, core::Point::new(-1, -1), 2, core::BORDER_CONSTANT, core::Scalar::default()).unwrap();
    let mut contours = VectorOfVectorOfPoint::new();
    imgproc::find_contours(&dilated, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0)).unwrap();
    let mut motion_detected = false;
    for i in 0..contours.len() {
        let area = imgproc::contour_area(&contours.get(i).unwrap(), false).unwrap();
        if area > 1000.0 {
            motion_detected = true;
        }
    }
    motion_detected
}

fn main() {
    let mut cam = VideoCapture::new(0, videoio::CAP_ANY).unwrap();
    assert!(cam.is_opened().unwrap(), "Unable to open default camera!");

    let mut frames_in_two_secs = vec![Mat::default(); 2];
    let mut i = 0;
    while highgui::wait_key(1000).unwrap() < 1 {
        cam.read(&mut frames_in_two_secs[i]).unwrap();
        i += 1;
        if i == 2 {
            if motion_detection(&frames_in_two_secs) {
                println!("Motion detected!");
                let mut img = Mat::default();
                imgproc::cvt_color(&frames_in_two_secs[1], &mut img, imgproc::COLOR_BGR2GRAY, 0).unwrap();
                let timestamp = chrono::Local::now().format("%Y-%m-%d-%H-%M-%S").to_string();
                let path = format!("./output/{}.jpg", timestamp);
                std::fs::create_dir_all("./output").unwrap();
                imwrite(&path, &img, &core::Vector::new()).unwrap();
            }
            i = 0;
        }
    }
}
