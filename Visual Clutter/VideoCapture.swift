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

        return true
    }

    // Starts the video capture session.
    public func start() {
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
        
        //TODO: set preference for hardware
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU // Use both CPU and GPU
        
        // Load the Core ML model
        let model = try! yolov8l(configuration: .init()).model
        //        print("model loaded")
            
        /// VNCoreMLModel
        let detector = try! VNCoreMLModel(for: model)
        detector.featureProvider = ThresholdProvider()
        
        
        
        // Retrieves Data From Results
        
        let request = VNCoreMLRequest(model: detector) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                print("Failed to get results or cast to VNRecognizedObjectObservation")
                return
            }
            
            
            for observation in results {
                //Extract the first label and its confidence
                if let topLabel = observation.labels.first {
                    if self.selected == topLabel.identifier && topLabel.confidence >= 0.5{
                        DispatchQueue.main.async {
                            
                            self.rect = self.scaleBoundingBox(observation.boundingBox)
                            
                            self.applyBlurOutsideRect(to: pixelBuffer, in: self.rect)
                            
                            
                        }
                        break
                    }
                    else{
                        DispatchQueue.main.async {
                            self.rect = .zero
                            self.processedImage = nil
                        
                        }
                    }
                }
            }
        }

        // Perform the Vision request on the pixel buffer (video frame)
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    private func scaleBoundingBox(_ boundingBox: CGRect) -> CGRect {
        let screenWidth = UIScreen.main.bounds.width
        let screenHeight = UIScreen.main.bounds.height
        let padding: CGFloat = 10.0
    

        let rect =  CGRect(
            x: (boundingBox.origin.x * screenWidth) ,
            y: (1 - boundingBox.origin.y - boundingBox.height) * screenHeight - 40, // Flip Y-axis
            width: boundingBox.width * screenWidth,
            height: boundingBox.height * screenHeight + (2 * padding)
        )
        
        return rect
        
        
    }
    
    
    func applyBlurOutsideRect(to pixelBuffer: CVPixelBuffer, in rect: CGRect) {
        // Print the input CGRect
        print("Input CGRect: \(rect)")

        // Convert CVPixelBuffer to CIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Scale the CIImage to match the CameraView's aspect ratio
        let viewSize = UIScreen.main.bounds.size
        let (scaledImage, scale) = scaleImageToMatchView(ciImage, viewSize: viewSize)
        
        // Adjust the CGRect for scaling
        let adjustedRect = adjustRectForScaling(rect, scale: scale)
        
        
        
        // Print the input image extent
        print("Input Image Extent: \(ciImage.extent)")

        // Apply a blur filter to the entire image
        let blurFilter = CIFilter(name: "CIGaussianBlur")
        blurFilter?.setValue(scaledImage, forKey: kCIInputImageKey)
        blurFilter?.setValue(5.0, forKey: kCIInputRadiusKey) // Adjust blur intensity
        
        // Get the blurred image
        guard let blurredImage = blurFilter?.outputImage else {
            print("Failed to apply blur filter.")
            return
        }

        // Create a mask for the rectangle region
        let maskFilter = CIFilter(name: "CICrop")
        maskFilter?.setValue(scaledImage, forKey: kCIInputImageKey)
        maskFilter?.setValue(CIVector(cgRect: adjustedRect), forKey: "inputRectangle")
        
        guard let maskedImage = maskFilter?.outputImage else {
            print("CICrop filter failed to produce an output image.")
            return
        }

        // Print the output image extent
        print("Output Image Extent: \(maskedImage.extent)")

        // Composite the original rectangle region back onto the blurred image
        let compositeFilter = CIFilter(name: "CISourceOverCompositing")
        compositeFilter?.setValue(maskedImage, forKey: kCIInputImageKey)
        compositeFilter?.setValue(blurredImage, forKey: kCIInputBackgroundImageKey)
        
        // Get the final output image
        guard let outputImage = compositeFilter?.outputImage else {
            print("Failed to composite the final image.")
            return
        }

        // Convert the CIImage to CGImage
        let context = CIContext()
        if let cgImage = context.createCGImage(outputImage, from: outputImage.extent) {
            DispatchQueue.main.async {
                self.processedImage = cgImage // Update the processed image
            }
        } else {
            print("Failed to convert CIImage to CGImage.")
        }
    }
    
    func scaleImageToMatchView(_ ciImage: CIImage, viewSize: CGSize) -> (CIImage, CGFloat) {
        let imageSize = ciImage.extent.size
        let scaleX = viewSize.width / imageSize.width
        let scaleY = viewSize.height / imageSize.height
        let scale = max(scaleX, scaleY) // Scale to fill the view
        
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return (scaledImage, scale)
    }
    
    func adjustRectForScaling(_ rect: CGRect, scale: CGFloat) -> CGRect {
        
        let xpadding: CGFloat = 60.0
        let ypadding: CGFloat = 50.0


        let myrect = CGRect(
            x: rect.origin.x * scale + xpadding,
            y: rect.origin.y * scale + ypadding,
            width: rect.width,
            height: rect.height
        )
        return myrect
    }
    
    
    
   
    
}
    
