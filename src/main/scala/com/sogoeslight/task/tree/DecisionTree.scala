package com.sogoeslight.task.tree

import com.sogoeslight.task.preparation.Data._

import scala.collection.mutable.ArrayBuffer

object DecisionTree extends App {
  def create(): Unit = {
    // values for calculating entropy
    val positive: Double = convertedData.count(_.last == 1.0)
    val negative: Double = convertedData.length - positive
    val total: Double = convertedData.length

    def log2: Double => Double = (x: Double) => math.log10(x) / math.log10(2.0)

    def calcEntropy(pos: Double, neg: Double, total: Double): Double = (pos, neg) match {
      case (0, 0) => 0
      case (_, 0) | (0, _) => 1
      case _ => -1 * (pos / total) * log2(pos / total) - (neg / total) * log2(neg / total)
    }

    val totalEntropy: Double = calcEntropy(positive, negative, total)

    // find averages of each attribute values
    def getDelimiters(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[Double] = {
      def findOne(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[Double] = {
        ds.transpose.map(_.sum / (cleanDataset.length - 1))
      }

      findOne(ds)
    }

    def calcInfoGains(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[BigDecimal] = {
      val delimiters: ArrayBuffer[Double] = getDelimiters(ds)
      val informationGains: ArrayBuffer[BigDecimal] = ArrayBuffer[BigDecimal]()

      def countOnes(moreOrLess: ArrayBuffer[ArrayBuffer[Double]]): Double = moreOrLess.count(p => p.last == 1.0)

      def countZeroes(moreOrLess: ArrayBuffer[ArrayBuffer[Double]]): Double = moreOrLess.count(p => p.last == 0)

      for (index <- 0 to ds(0).size - 2) {
        val moreThanDelimiter: ArrayBuffer[ArrayBuffer[Double]] = ds.filter(_ (index) >= delimiters(index))
        val attrMoreEntropy: Double = calcEntropy(countOnes(moreThanDelimiter), countZeroes(moreThanDelimiter), moreThanDelimiter.size)

        val lessThanDelimiter: ArrayBuffer[ArrayBuffer[Double]] = ds.filter(_ (index) < delimiters(index))
        val attrLessEntropy: Double = calcEntropy(countOnes(lessThanDelimiter), countZeroes(lessThanDelimiter), lessThanDelimiter.size)

        informationGains += totalEntropy - (moreThanDelimiter.size / total) * attrMoreEntropy - (lessThanDelimiter.size / total) * attrLessEntropy
      }

      informationGains
    }

    def getMaxInfoGain(IGs: ArrayBuffer[BigDecimal]): Int = IGs.indexOf(IGs.max)

    def splitNode(ds: ArrayBuffer[ArrayBuffer[Double]], bestAttributeIndex: Int, del: Double): (Node, Node) = {
      val (leftChild, rightChild) = (Node(), Node())

      for (row <- ds) {
        val removed: Double = row.remove(bestAttributeIndex)
        if (removed >= del) leftChild.dataset += row else rightChild.dataset += row
      }

      (leftChild, rightChild)
    }

    def mostCommonClass(ds: ArrayBuffer[ArrayBuffer[Double]]): String = {
      val negatives: Int = ds.count(_.last == 0)
      val positives: Int = ds.count(_.last == 1)
      if (positives > negatives) "1"
      else "0"
    }

    /**
     * Main algorithm for decision tree creating
     *
     * @param ds            - dataset
     * @param attributeList - list of attributes
     * @return Tree
     */
    def id3train(ds: ArrayBuffer[ArrayBuffer[Double]], attributeList: Array[String]): Node = {
      val root: Node = Node()
      val dsToTrain: ArrayBuffer[ArrayBuffer[Double]] = ds.to(ArrayBuffer).map(_.to(ArrayBuffer))
      val attributes: ArrayBuffer[String] = attributeList.to(ArrayBuffer)

      val classColumn: ArrayBuffer[Double] = dsToTrain.transpose.last
      if (!classColumn.contains(0)) { // check if all examples are positive
        root.label = "1"
      } else if (!classColumn.contains(1)) { // check if all examples are negative
        root.label = "0"
      } else if (attributes.isEmpty) { // check if any attributes left
        val zeroes: Int = classColumn.count(_ == 0)
        val one: Int = classColumn.count(_ == 1)
        root.label = (zeroes max one).toString
      } else {
        val bestAttributeIndex = getMaxInfoGain(calcInfoGains(dsToTrain))
        root.bestAttributeIndex = Some(bestAttributeIndex)
        root.label = attributes(bestAttributeIndex)
        root.delimiter = getDelimiters(dsToTrain)(bestAttributeIndex)
        root.dataset = dsToTrain

        val (left, right) = splitNode(dsToTrain, bestAttributeIndex, root.delimiter)
        root.left = left
        root.left.dataset = left.dataset
        root.right = right
        root.right.dataset = right.dataset

        val updateAttributesList: ArrayBuffer[String] = attributes -= attributes(bestAttributeIndex)

        def newNode(node: Node, attrList: ArrayBuffer[String], isLeft: Boolean): Unit = {
          if (node.dataset.isEmpty || attrList.isEmpty) {
            node.label = mostCommonClass(node.dataset)
          } else {
            if (isLeft) root.left = id3train(root.left.dataset, updateAttributesList.toArray)
            else root.right = id3train(root.right.dataset, updateAttributesList.toArray)
          }
        }

        newNode(left, attributes, isLeft = true)
        newNode(right, attributes, isLeft = false)
      }
      root
    }

    /**
     *
     * @param tree tree for testing
     * @param ds   testing dataset
     * @return accuracy in %
     */
    def testTree(tree: Node, ds: Array[Array[Double]]): Double = {
      def check(node: Node, ds: Array[Array[Double]]): Double = {
        var correctAnswersCounter: Int = 0

        for (row <- ds) {
          val expected: String = row.last.toInt.toString
          var testNode: Node = node
          while (testNode.bestAttributeIndex.isDefined) {
            if (row(testNode.bestAttributeIndex.get) < testNode.delimiter) testNode = testNode.right
            else testNode = testNode.left
          }
          val realValue: String = testNode.label
          if (expected == realValue) correctAnswersCounter += 1
        }

        correctAnswersCounter
      }

      check(tree, ds) / ds.length * 100
    }

    def printTreeAndAllTheData(): Unit = {
      val tree: Node = id3train(learning.to(ArrayBuffer).map(_.to(ArrayBuffer)), attributes)

      println("\nLAB 2 - Decision Tree\n")
      tree.draw()
      println()
      tree.printDepth()
      tree.printAmountOfLeaves()

      println("- Metrics: Information gain")
      println(f"- Accuracy for test (${test.length} rows) array: ${testTree(tree, test)}%.2f%%")
      println(f"- Accuracy for learning (${learning.length} rows) array: ${testTree(tree, learning)}%.2f%%")
      println("\n____________________________________________________________________________________")
    }

    printTreeAndAllTheData()
  }
}
