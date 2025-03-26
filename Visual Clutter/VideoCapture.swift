//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Video Capture for Ultralytics YOLOv8 Preview on iOS
//  Part of the Ultralytics YOLO app, this file defines the VideoCapture class to interface with the device's camera,
//  facilitating real-time video capture and frame processing for YOLOv8 model previews.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This class encapsulates camera initialization, session management, and frame capture delegate callbacks.
//  It dynamically selects the best available camera device, configures video input and output, and manages
//  the capture session. It also provides methods to start and stop video capture and delivers captured frames
//  to a delegate implementing the VideoCaptureDelegate protocol.

// The code bellow was altered based on the ultralytics code.

import AVFoundation
import CoreVideo
import UIKit
import Vision

// Defines the protocol for handling video frame capture events.
public protocol VideoCaptureDelegate: AnyObject {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame: CMSampleBuffer)
}

// Identifies the best available camera device based on user preferences and device capabilities.
func bestCaptureDevice() -> AVCaptureDevice {
  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: .back)
  {
    return device
  } else if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: .back)
  {
    return device
  } else {
    fatalError("Expected back camera device is not available.")
  }
}

public class VideoCapture: NSObject,ObservableObject {
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?
    
    @Published var rect: CGRect = .zero // Store the bounding box for object detection
    @Published var label:String = "cup"
    @Published var selected:String = "cup"
    @Published var processedImage: CGImage?
    @Published var shouldApplyBlur = false


    let captureDevice = bestCaptureDevice()
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    var cameraOutput = AVCapturePhotoOutput()
    let queue = DispatchQueue(label: "camera-queue")

    // Configures the camera and capture session with optional session presets.
    public func setUp(
        sessionPreset: AVCaptureSession.Preset = .hd1280x720, completion: @escaping (Bool) -> Void
    ) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }

    // Internal method to configure camera inputs, outputs, and session properties.
    private func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset

        // Setup video input
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            return false
        }

        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }

        // Setup video preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
        self.previewLayer = previewLayer

        // Setup video output
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]

        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Setup camera output (photo output)
        if captureSession.canAddOutput(cameraOutput) {
            captureSession.addOutput(cameraOutput)
        }

        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        do {
            try captureDevice.lockForConfiguration()
            captureDevice.focusMode = .continuousAutoFocus
            captureDevice.exposureMode = .continuousAutoExposure
            captureDevice.unlockForConfiguration()
        } catch {
            print("Unable to configure the capture device.")
            return false
        }

        // Commit configuration to finalize setup
        captureSession.commitConfiguration()
        captureSession.startRunning()

        return true
    }

    // Starts the video capture session.
    public func start() {
        self.rect = .zero
        queue.async {
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
                DispatchQueue.main.async {
                    print("Camera session started.")
                }
            }
        }
    }

    // Stops the video capture session.
    public func stop() {
        self.rect = .zero
        if captureSession.isRunning {
            captureSession.stopRunning()
                    
            for output in captureSession.outputs{
                captureSession.removeOutput(output)
            }
                        
            
        }
    }

    func updateVideoOrientation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        switch UIDevice.current.orientation {
        case .portrait:
            connection.videoOrientation = .portrait
        case .portraitUpsideDown:
            connection.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            connection.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            connection.videoOrientation = .landscapeRight
        default:
            return
        }
        self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    }
}

// Extension to handle AVCaptureVideoDataOutputSampleBufferDelegate events.
extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        delegate?.videoCapture(self, didCaptureVideoFrame: sampleBuffer)
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        
        // Pass the pixel buffer (video frame) to the Core ML model using Vision
        processFrame(pixelBuffer: pixelBuffer)
    }
    
    
    
    
    public func captureOutput(
        _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Optionally handle dropped frames, e.g., due to full buffer.
    }
    
    // Method to process the frame using Vision and Core ML
    func processFrame(pixelBuffer: CVPixelBuffer) {
        
        // Convert CVPixelBuffer to CIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Resize the image to [384, 640]
        let resizedImage = resizeImage(ciImage, to: CGSize(width: 384, height: 640))

        // Convert the resized CIImage back to CVPixelBuffer
        guard let resizedPixelBuffer = resizedImage.toPixelBuffer() else {
            print("Failed to convert resized CIImage to CVPixelBuffer.")
            return
        }
        
        //TODO: set preference for hardware
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU // Use both CPU and GPU
        
        // Load the Core ML model
        let model = try! yolov8m(configuration: .init()).model
        //        print("model loaded")
        
        /// VNCoreMLModel
        let detector = try! VNCoreMLModel(for: model)
        detector.featureProvider = ThresholdProvider()
        
        
        
//        // Retrieves Data From Results
//
//        let request = VNCoreMLRequest(model: detector) { request, error in
//            guard let results = request.results as? [VNRecognizedObjectObservation] else {
//                print("Failed to get results or cast to VNRecognizedObjectObservation")
//                return
//            }
//            
//            
//            for observation in results {
//                //Extract the first label and its confidence
//                if let topLabel = observation.labels.first {
//                    if self.selected == topLabel.identifier && topLabel.confidence >= 0.5{
//                        DispatchQueue.main.async {
//                            
//                            self.rect = self.scaleBoundingBox(observation.boundingBox)
//                            
//                            
//                        }
//                        break
//                    }
//                    else{
//                        DispatchQueue.main.async {
//                            self.rect = .zero
//                            self.processedImage = nil
//                        
//                        }
//                    }
//                }
//            }
//        }
        
        // Create the Vision request
        let request = VNCoreMLRequest(model: detector) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation],
                  let bestObservation = results.max(by: { $0.confidence < $1.confidence }) // Get highest confidence
            else {
                DispatchQueue.main.async {
                    self.rect = .zero
                    self.processedImage = nil
                }
                return
            }
            
            if let topLabel = bestObservation.labels.first,
               self.selected == topLabel.identifier, topLabel.confidence >= 0.5 {

                DispatchQueue.main.async {
                    self.rect = self.scaleBoundingBox(bestObservation.boundingBox)
                }
            } else {
                DispatchQueue.main.async {
                    self.rect = .zero
                    self.processedImage = nil
                }
            }
        }

        // Perform the Vision request on the pixel buffer (video frame)
        let handler = VNImageRequestHandler(cvPixelBuffer: resizedPixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    private func scaleBoundingBox(_ boundingBox: CGRect) -> CGRect {
        let screenWidth = UIScreen.main.bounds.width
        let screenHeight = UIScreen.main.bounds.height
        let padding: CGFloat = 10.0
    

        let rect =  CGRect(
            x: (boundingBox.origin.x * screenWidth-20) ,
            y: (1 - boundingBox.origin.y - boundingBox.height) * screenHeight-50, // Flip Y-axis
            width: boundingBox.width * screenWidth + (2 * padding),
            height: boundingBox.height * screenHeight + (2 * padding)
        )
        
        return rect
        
        
    }
    
    // Helper function to resize CIImage
    func resizeImage(_ image: CIImage, to size: CGSize) -> CIImage {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height
        return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }
    
}
