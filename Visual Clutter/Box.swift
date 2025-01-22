//
//  Box.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 24/10/2024.
//

import Foundation
import UIKit
import SwiftUI

//struct BoundingBoxView: View {
//    var rect: CGRect
//    
//    var body: some View {
//        Rectangle()
//            .stroke(Color.red, lineWidth: 2) // Draws a red outline
//            .frame(width: rect.width, height: rect.height)
//            .position(x: rect.midX, y: rect.midY) // Center the rectangle at the specified position
//    }
//}


struct BoundingBoxViewRepresentable: UIViewRepresentable {
    var boundingBoxes: [CGRect]
    
    func makeUIView(context: Context) -> BoundingBoxView {
        return BoundingBoxView()
    }
    
    func updateUIView(_ uiView: BoundingBoxView, context: Context) {
        uiView.boundingBoxes = boundingBoxes
        uiView.setNeedsDisplay() // Refresh the view to show updated bounding boxes
    }
}


class BoundingBoxView: UIView {
    var boundingBoxes: [CGRect] = []
    
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        context.setStrokeColor(UIColor.red.cgColor)
        context.setLineWidth(2.0)
        
        for box in boundingBoxes {
            context.stroke(box)
        }
    }
}

// Function to convert bounding box output to CGRect
func convertToCGRect(boundingBox: [Float]) -> CGRect {
    let x = CGFloat(boundingBox[0])
    let y = CGFloat(boundingBox[1])
    let width = CGFloat(boundingBox[2])
    let height = CGFloat(boundingBox[3])
    return CGRect(x: x, y: y, width: width, height: height)
}
//
//// Example function to display the image and bounding boxes
//func displayImageWithBoundingBoxes(image: UIImage, boxes: [[Float]]) {
//    let imageView = UIImageView(image: image)
//    imageView.contentMode = .scaleAspectFit
//
//    // Create a BoundingBoxView to draw the boxes
//    let boundingBoxView = BoundingBoxView(frame: imageView.bounds)
//    
//    // Convert bounding box data to CGRects
//    for box in boxes {
//        let rect = convertToCGRect(boundingBox: box, imageSize: image.size)
//        print(rect)
//        boundingBoxView.boundingBoxes.append(rect)
//    }
//    
//    // Add the image view and bounding box view to a parent view
//    let parentView = UIView(frame: CGRect(origin: .zero, size: image.size))
//    parentView.addSubview(imageView)
//    parentView.addSubview(boundingBoxView)
//
//    // Make sure the bounding box view is on top
//    boundingBoxView.backgroundColor = .clear
//    boundingBoxView.isUserInteractionEnabled = false // Allow interactions on the image view
//}
