var slideIndex = 1;
showDivs(slideIndex);

function plusDivs(n) {
    showDivs(slideIndex += n);
}

function showDivs(n) {
    var i;
    var x = document.getElementsByClassName("mySlides");
    if (n > x.length) {slideIndex = 1}
    if (n < 1) {slideIndex = x.length} ;
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    x[slideIndex-1].style.display = "block";
}

/* Now try my way... */
function display(pic_index) {
  var x = document.getElementsByClassName("mySlides");
  /* Must remove the inactive one too, hence the crazyness */
  /* For two photos... */
  if (x.length == pic_index + 1) {
    x[pic_index-1].style.display = "none";
    x[pic_index].style.display = "block";
    x[pic_index].style.visibility = "visible";
  } else {
    x[pic_index+1].style.display = "none";
    x[pic_index].style.display = "block";
    x[pic_index].style.visibility = "visible";
  }
}
