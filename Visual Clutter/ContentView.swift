//
//  ContentView.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 2/10/2024.
//

import SwiftUI
import CoreMedia
import AVFoundation

// UIViewRepresentable to bridge UIKit's AVCaptureVideoPreviewLayer to SwiftUI
struct CameraView: UIViewRepresentable {
    let videoCapture: VideoCapture

    func makeCoordinator() -> Coordinator {
        Coordinator(videoCapture: videoCapture)
    }

    class Coordinator {
        var videoCapture: VideoCapture
        
        init(videoCapture: VideoCapture) {
            self.videoCapture = videoCapture
        }
    }
    
    // Creates the UIView to display the camera preview
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        
        // Set up the camera session
        videoCapture.setUp { success in
            if success {
                DispatchQueue.main.async {
                    if let previewLayer = self.videoCapture.previewLayer {
                        previewLayer.frame = view.bounds
                        view.layer.addSublayer(previewLayer)
                        print("Preview layer added")
                    } else {
                        print("No preview layer available.")
                    }
                }
            } else {
                print("Failed to set up camera.")
            }
        }
        
        return view
    }
    
    // Updates the UIView when the SwiftUI view state changes
    func updateUIView(_ uiView: UIView, context: Context) {
        videoCapture.previewLayer?.frame = uiView.bounds
    }
    
    // Camera session starts and stops are now handled in ContentView
}

// SwiftUI view containing the camera preview
struct ContentView: View {
    let videoCapture = VideoCapture()
    
    var body: some View {
        ZStack{
            CameraView(videoCapture: videoCapture)
            .onAppear {
                print("Camera view appeared")
                requestCameraPermission {
                    self.videoCapture.start()
                    self.videoCapture.setUp()
                }
            }
            .onDisappear {
                print("Camera view disappeared")
                self.videoCapture.stop()
            }
            .edgesIgnoringSafeArea(.all)

            VStack{
                Spacer()
                Button(action: loadModel, label: {
                    Text ("Start Model")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(width: 200 ,height: 50)
                        .background(Color.gray)
                        .cornerRadius(25)
                })
                .padding(.bottom,50)
                
            }
        }
    }

    // Helper function to request camera permission
    func requestCameraPermission(completion: @escaping () -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    DispatchQueue.main.async {
                        completion()
                    }
                } else {
                    print("Camera access denied.")
                }
            }
        case .denied, .restricted:
            print("Camera access restricted or denied.")
        @unknown default:
            print("Unknown camera authorization status.")
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
        // Pass the pixel buffer (video frame) to the Core ML model using Vision
        processFrame(pixelBuffer: pixelBuffer)
    }

    // Method to process the frame using Vision and Core ML
    func processFrame(pixelBuffer: CVPixelBuffer) {
        // Load the Core ML model
        let model = try! yolov8n(configuration: .init()).model
    
        // Create a Vision request with the Core ML model
        let request = VNCoreMLRequest(model: model) { request, error in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                // Handle detected objects here
                self.handleDetections(results)
            }
        }
    
        // Perform the Vision request on the pixel buffer (video frame)
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }

    func handleDetections(_ results: [VNRecognizedObjectObservation]) {
        var newBoxes: [CGRect] = []
        var newLabels: [String] = []

        print(results)
        
        // for result in results {
        //     // Get the bounding box and label for the detected object
        //     let boundingBox = result.boundingBox
        //     let label = result.labels.first?.identifier ?? "Unknown"
            
        //     // Convert bounding box to screen coordinates (as shown earlier)
        //     let screenWidth = previewLayer.frame.width
        //     let screenHeight = previewLayer.frame.height
        //     let x = boundingBox.origin.x * screenWidth
        //     let y = (1 - boundingBox.origin.y - boundingBox.height) * screenHeight
        //     let width = boundingBox.width * screenWidth
        //     let height = boundingBox.height * screenHeight
            
        //     newBoxes.append(CGRect(x: x, y: y, width: width, height: height))
        //     newLabels.append(label)
        // }
        
        // // Update UI with the new bounding boxes and labels
        // DispatchQueue.main.async {
        //     self.boxes = newBoxes
        //     self.labels = newLabels
        // }
    }

    func loadModel(){
        print("model loading")
    }
}
