package com.sogoeslight.task.preparation

import scala.collection.mutable.ArrayBuffer
import scala.io.StdIn.readLine
import scala.io.{BufferedSource, Source}
import scala.util.Random

object Data extends App {

  final lazy val datasetRegex = ","
  final lazy val path_to_data: String = "./src/main/resources/data.txt"

  final lazy val dataset: BufferedSource = Source.fromFile(path_to_data)
  final lazy val parsedDataset: ArrayBuffer[ArrayBuffer[String]] = dataset.getLines()
    .map(_.split(datasetRegex).to(ArrayBuffer)).to(ArrayBuffer)
  dataset.close()

  final lazy val attributes: Array[String] = parsedDataset.head.dropRight(1).toArray
  final lazy val attributeCount: Option[Int] = Some(attributes.length)

  final lazy val cleanDataset: Array[Array[String]] = {
    var cleanData: ArrayBuffer[ArrayBuffer[String]] = ArrayBuffer[ArrayBuffer[String]]()

    for (row <- parsedDataset) { // removing extra spaces and empty fields
      val cleanRow: ArrayBuffer[String] = row.map(_.trim).filter(_.nonEmpty)
      if (cleanRow.size == attributeCount.get + 1) cleanData += cleanRow
    }

    cleanData.toArray.map(_.toArray)
  }

  final lazy val rowsCount: Option[Int] = Some(parsedDataset.dropRight(1).length) // - attributes row
  final lazy val cleanRowsCount: Option[Int] = Some(cleanDataset.dropRight(1).length) // - attributes row

  final lazy val categoricalColumnIndexes = List() // <-- ADD HERE indexes of columns in case you have categorical attributes
  final lazy val binData: Array[Array[String]] = Converter.catToBin(cleanDataset, categoricalColumnIndexes)
  final lazy val convertedData: Array[Array[Double]] = binData.tail.map(_.map(_.toDouble))
  final lazy val classColumn: Array[Double] = convertedData.transpose.last

  lazy val shuffledSamples: Array[Array[Double]] = Random.shuffle(convertedData.to(ArrayBuffer).map(_.map(identity))).toArray
  lazy val (learning, test) = shuffledSamples.splitAt(math.ceil(cleanRowsCount.get * 0.75).toInt)

  def prepare(): Unit = {
    println("\nLAB 1 - Data Preparation")
    println(s"\n- Rows count: ${rowsCount.getOrElse("No rows found")}")
    println(s"- Attribute count: ${attributeCount.getOrElse("No attributes found")}")
    println(s"- Clean data rows count: ${cleanRowsCount.getOrElse("No rows found")}\n")

    def printFirstFiveRows(): Unit = {
      println(s"- First 5 rows: ")
      cleanDataset.take(6).foreach(x => {
        println("    " + x.mkString(" | "))
      })
    }

    printFirstFiveRows()

    println(s"\n- Learning samples: ${learning.length}")
    println(s"\n- Test samples: ${test.length}")

    readLine("\nPress enter to show learning, test samples and categorical to binary converted data...")

    def printSamples[A](sample: Array[Array[A]], name: String, isWithHeader: Boolean): Unit = {
      println(s"${name.toUpperCase}:")
      if (isWithHeader) {
        println(convertedData(0).mkString(" | "))
        sample.tail.zipWithIndex.foreach { case (x, index) => println(s"${index + 1}) ${x.mkString(" | ")}") }
      } else {
        sample.zipWithIndex.foreach { case (x, index) => println(s"${index + 1}) ${x.mkString(" | ")}") }
      }
    }

    printSamples(learning, "learning samples", isWithHeader = false)
    printSamples(test, "test samples", isWithHeader = false)
    if (binData sameElements cleanDataset) println("\nNo categorical attributes, dataset was not converted")
    else printSamples(convertedData, "categorical to binary converted data", isWithHeader = true)

    println("\n____________________________________________________________________________________")
  }

  prepare()
}
