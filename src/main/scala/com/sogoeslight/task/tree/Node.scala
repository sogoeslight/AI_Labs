package com.sogoeslight.task.tree

import scala.collection.mutable.ArrayBuffer

case class Node(var dataset: ArrayBuffer[ArrayBuffer[Double]] = ArrayBuffer[ArrayBuffer[Double]](),
                var attributes: ArrayBuffer[String] = ArrayBuffer[String](),
                var bestAttributeIndex: Option[Int] = None,
                var delimiter: Double = 0,
                var label: String = "",
                var left: Node = null,
                var right: Node = null,
                var disp: Double = 0,
                var leafAvg: Double = 0
               ) {

  def printDepth(): Unit = {
    println(s"- Tree depth: ${maxDepth(root = this)}")
  }

  def maxDepth(root: Node): Int = root match {
    case null => 0
    case _ => math.max(maxDepth(root.left), maxDepth(root.right)) + 1
  }

  def printAmountOfLeaves(): Unit = {
    println(s"- Leaves: ${leaves(root = this)}")
  }

  def leaves(root: Node): Int = root match {
    case null => 0
    case node if node.left == null && node.right == null => 1
    case _ => leaves(root.right) + leaves(root.left)
  }

  def draw(): Unit = {
    draw(prefix = "", node = this, left = false)
  }

  def draw(prefix: String, node: Node, left: Boolean): Unit = {
    if (node != null) {
      if (node.delimiter != 0) {
        println(prefix + (
          if (left) f"|---- ${node.label}, < ${node.delimiter}%.7f"
          else f"\\---- ${node.label}, > ${node.delimiter}%.7f")
        )
      } else {
        println(prefix + (
          if (left) f"|---- "
          else f"\\---- ")
          + node.label
        )
      }
      draw(prefix + (if (left) "|   " else "    "), node.left, left = true)
      draw(prefix + (if (left) "|   " else "    "), node.right, left = false)
    }
  }
}
