//
//  ContentView.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 2/10/2024.
//

import SwiftUI
import CoreMedia
import AVFoundation
import Vision


// UIViewRepresentable to bridge UIKit's AVCaptureVideoPreviewLayer to SwiftUI
struct CameraView: UIViewRepresentable {
    let videoCapture: VideoCapture
    
    @Binding var processedImage: CGImage? // Binding to receive the processed frame


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
    
    func updateUIView(_ uiView: UIView, context: Context) {
            // Update the preview layer's frame
            videoCapture.previewLayer?.frame = uiView.bounds
            
            // If a processed image is available, display it
            if let processedImage = processedImage {
                // Remove previous layers
                uiView.layer.sublayers?.removeAll()
                
                // Create a new layer for the processed image
                let imageLayer = CALayer()
                imageLayer.contents = processedImage
                
                // Scale the image layer to fill the view
                imageLayer.frame = uiView.bounds
                imageLayer.contentsGravity = .resizeAspectFill // Maintain aspect ratio while filling the view
                
                // Add the image layer to the view
                uiView.layer.addSublayer(imageLayer)
            } else {
                // If no processed image, ensure only the live feed is displayed
                uiView.layer.sublayers?.removeAll()
                if let previewLayer = videoCapture.previewLayer {
                    uiView.layer.addSublayer(previewLayer)
                }
            }
        }
    

}

class BoundingBoxManager: ObservableObject {
    @Published var boundingBoxes: [CGRect] = []
}

// SwiftUI view containing the camera preview
struct ContentView: View {
    
    @ObservedObject var videoCapture = VideoCapture()
    let items = ["cup", "bottle", "knife", "spoon", "laptop", "scissors"]
    
    @State private var selectedItem: String? = nil
    @State private var isMenuOpen = false
    @State private var modelStatus = false
    
    @State private var scanTimeRemaining = 10
    @State private var timer: Timer?
    @State private var showTimeoutMessage = false



    
    
    var body: some View {
        ZStack{
            if modelStatus{
                CameraView(videoCapture: videoCapture, processedImage: $videoCapture.processedImage)
                    .edgesIgnoringSafeArea(.all)
                
                
                // Add additional UI elements only when processedImage is nil
                if videoCapture.processedImage == nil {
                    VStack {
                        if showTimeoutMessage {
                            Text("Unable to find \(videoCapture.selected) on the current surface. Please try another one.")
                                .foregroundColor(.white)
                                .font(.headline)
                                .multilineTextAlignment(.center)
                                .padding()
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(10)
                                .padding(.top, 50)
                        } else {
                            Text("No \(videoCapture.selected) detected. Please move the camera slowly to scan the area.")
                                .foregroundColor(.white)
                                .font(.headline)
                                .multilineTextAlignment(.center)
                                .padding()
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(10)
                                .padding(.top, 50)
                        }

                        Spacer() // Push the text to the top
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity) // Ensure the VStack takes up the full screen
                    .onAppear {
                        startScanTimer()
                    }
                }
                        

            }else{
                
                Color.white
                    .ignoresSafeArea(.all)
                
                Text("CVI Clutter Tool Kit")
                    .font(.system(size: 30, weight: .bold, design: .rounded))
                    .foregroundColor(.black) // Neutral text color
                    .padding(15) // Add padding inside the border
                    .background(
                        RoundedRectangle(cornerRadius: 20) // Rounded border
                            .fill(Color.white) // White background
                            .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                            )
                    )
                
                VStack {
                    HStack {
                        Text("MATT")
                            .font(.system(size: 12, weight: .bold, design: .rounded))
                            .foregroundColor(.black) // Neutral text color
                            .padding(15) // Add padding inside the border
                            .background(
                                RoundedRectangle(cornerRadius: 20) // Rounded border
                                    .fill(Color.white) // White background
                                    .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 20)
                                            .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                                    )
                            )
                        
                            .padding(.horizontal,20)

                    
                        Spacer() // Push the text to the left
                    }
                    Spacer() // Push the HStack to the top
                }
                
            }
            
            
            if modelStatus{
                Rectangle()
                    .stroke(Color.red, lineWidth: 2)
                    .frame(width: videoCapture.rect.width,
                           height: videoCapture.rect.height)
                    .position(x: videoCapture.rect.midX ,
                              y: (videoCapture.rect.midY)) // Flip Y-axis for Vision bounding box
            }
            
        
            VStack(spacing: 20) {
                
                Spacer()
                
                Button(action: {
                    if modelStatus {
                        self.videoCapture.stop()
                    }
                    else{
                        
                        self.videoCapture.start()
                    }
                    modelStatus.toggle()
                }, label: {
                    Text(modelStatus ? "Stop Looking For..." : "Start Looking For...")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(width: 200, height: 50)
                        .background(Color(hex: "#355070"))
                        .cornerRadius(25)
                })
                
                Button(action: {
                    isMenuOpen.toggle() // Toggle the menu state
                }, label: {
                    HStack {
                        Text(videoCapture.selected.capitalized)
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        // Add an icon (e.g., a chevron down) next to the text
                        Image(systemName: isMenuOpen ? "chevron.up" : "chevron.down") // Change icon based on menu state
                            .foregroundColor(.white)
                            .font(.system(size:12 ,weight: .black))
                    }
                    .padding()
                    .frame(width: 200, height: 50)
                    .background(Color(hex: "#818589"))
                    .cornerRadius(25)
                })
                
            
            }
            

            
            if isMenuOpen{
                                
                VStack {
                    
//                    Spacer().frame(height: 50)
                    
//                    Text("Select Item For Detection")
//                        .font(.headline)
                        

                    ScrollView {
                        
                        LazyVGrid(
                            columns: [
                                GridItem(.flexible(), spacing: 16), // First column
                                GridItem(.flexible(), spacing: 16)  // Second column
                            ],
                            spacing: 16 // Vertical spacing between rows
                        ) {
                            ForEach(items, id: \.self) { item in
                                Button(action: {
                                    chooseItem(item: item)
                                    isMenuOpen = false
                                }) {
                                    Text(item.capitalized)
                                        .font(.body)
                                        .fontWeight(.semibold)
                                        .padding()
                                        .frame(maxWidth: .infinity)
                                        .background(
                                            RoundedRectangle(cornerRadius: 20) // Rounded border
                                                .fill(Color.white) // White background
                                                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 20)
                                                        .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                                                )
                                        )
                                        .cornerRadius(8)
                                        .foregroundColor(Color.black)
                                }
                                .padding(.horizontal, 16) // Adjust horizontal padding inside each grid cell
                            }
                        }
                        .padding(16)
                    }
                    .background(Color.white)
                    .cornerRadius(12)
                    .shadow(radius: 5)
                    .frame(maxHeight: 225)
                    .padding(.horizontal,10)
                }
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

    // Loads the model (activated by the start button)
    func loadModel(){
        requestCameraPermission {
            self.videoCapture.start()
        }
    }
    
    // alters the current item and sends it to videocapture for processing. (Used by each button on the object menu)
    func chooseItem(item:String){
        videoCapture.selected = item
                
    }
    
    func startScanTimer() {
        scanTimeRemaining = 10
        showTimeoutMessage = false
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            if scanTimeRemaining > 0 {
                scanTimeRemaining -= 1
            } else {
                timer?.invalidate()
                // Show the timeout message after 10 seconds
                showTimeoutMessage = true
            }
        }
    }
    
}
