$(document).ready(function(){
	//INDEX.HTML
	$('#loading').hide();
	$('#inddata').hide();
	$('#csvdata').hide();


	$("#getstarted").click(function(event){
		$('#index-text').fadeOut( "slow", function() {
		});
		$('#index-img').fadeOut( "slow", function() {
			$('#index-form').fadeIn( "slow", function() {
			});
		});
	});

	$('#file').change(function() {
		$('#file-form').submit(); 
		$('#loading').show();
	});
	//RESULTS.HTML
	$("#analysisout").click(function(event){
		$('#miniinfo').fadeToggle();
	});

	$('#file').change(function() {
		$('#file-form').submit();
		$('#loading').show();
	});

	$("#continue").click(function(event){
		if(!$('.target').is(':checked'))
		{
			alert("Please, select the analysis output by clicking on a column's name.");
		}
	});

	//CONFIRM.HTML
	$("#left").click(function(event){
		$('#loading').show();
	});

	//TRAIN.HTML
	$(".manually").click(function(event){
		$('#inddata').show();
	});

	$("#continue4").click(function(event){
		$('#inddata').hide();
		$('#loading').show();
	});

	$('iframe.iframe1').load(function() {
        $('iframe.iframe1').contents().find(".lime.table_div").hide();
        $('iframe.iframe1').contents().find(".lime.top_div").css({"display":"flex","text-align":"center", 
        	"justify-content":"space-around","flex-wrap":"wrap","padding":"0 3%"});
	});

	$(".closei").click(function(event){
		$('#inddata').hide();
	});

	$(".closea").click(function(event){
		$('#A').hide();
		$('#loading').hide();
	});

	$(".closeb").click(function(event){
		$('#B').hide();
		$('#loading').hide();

	});
}); 

//GRAY - IGNORE
function reply_click(obj)
{
	//Toggle gray	
	$(obj).parent()[0].classList.toggle("colG");
    //If target not disabled, disable
    if( $(obj).parent().find(".target")[0].disabled == false){
    	$(obj).parent().find(".target")[0].disabled = true;
    }
	//Enable target if it was disabled
	else{
		$(obj).parent().find(".target")[0].disabled = false;
	}
}

//TARGET
function reply_clickT(obj)
{
	//Ensure there is only one target animation selected	
	if ($(".colT")[0]){
		//Enable switch if no longer target but ignore click from ignored columns
		if( $(obj).parent().find(".target")[0].disabled == false){
			$(".colT")[0].children[1].children[0].disabled = false;
			$(".colT").toggleClass("colT");
		}
	}
		//Only can be targeted when not ignored
		if( $(obj).parent().find(".target")[0].disabled == false){
			$(obj).parent()[0].classList.toggle("colT");
    	//Disable switch
    	$(obj).parent().find(".ignore")[0].disabled = true;
    }
}

$.fn.attachDragger = function(){
	var attachment = false, lastPosition, position, difference;
	$( $(this).selector ).on("mousedown mouseup mousemove",function(e){
		if( e.type == "mousedown" ) attachment = true, lastPosition = [e.clientX, e.clientY];
		if( e.type == "mouseup" ) attachment = false;
		if( e.type == "mousemove" && attachment == true ){
			position = [e.clientX, e.clientY];
			difference = [ (position[0]-lastPosition[0]), (position[1]-lastPosition[1]) ];
			$(this).scrollLeft( $(this).scrollLeft() - difference[0] );
            //$(this).scrollTop( $(this).scrollTop() - difference[1] );
            lastPosition = [e.clientX, e.clientY];
        }
    });
	$(window).on("mouseup", function(){
		attachment = false;
	});
}

$(document).ready(function(){
	$("#samples").attachDragger();
});

window.onclick = function(event) {
    if (event.target == document.getElementById('inddata')) {
        document.getElementById('inddata').style.display = "none";
    }
    if (event.target == document.getElementById('A')) {
        document.getElementById('A').style.display = "none";
        document.getElementById('loading').style.display = "none";

    }
    if (event.target == document.getElementById('B')) {
        document.getElementById('B').style.display = "none";
        document.getElementById('loading').style.display = "none";

    }
}

