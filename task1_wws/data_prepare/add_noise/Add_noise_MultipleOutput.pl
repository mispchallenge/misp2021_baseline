use strict;
use File::Path;
use File::Basename;

my $ROOT_DIR   = "add_noise";
my $CONFIG_DIR = "$ROOT_DIR/Config";
my $BIN_DIR    = "add_noise/bin";

my $CLEAN_WAV_DIR = "positive_reverb";
my $NOISY_WAV_DIR = "positive_addnoise";

my $CLEAN_WAV_SCP = "add_noise/scp/positive.scp";
my $NOISE_DATA_SCP = "add_noise/middleNoise.scp";
my @snrs          = (-15, -10, -5, 0, 5, 10, 15);
my $output_type = 1;

open(IN, $NOISE_DATA_SCP) || die "Error: Can't open noise_data scp: $NOISE_DATA_SCP\n";
my @noise_data = <IN>;
map chomp, @noise_data;
my $nNoise = scalar @noise_data;

::SplitFile($CLEAN_WAV_SCP, $nNoise);

for(my $i=1; $i<=@noise_data; $i++)
{
	for(my $j=0; $j<@snrs; $j++)
	{
		my $k = $i;
		my $noise_file = $noise_data[$i-1];
		my $clean_wav_scp_cur = "$CLEAN_WAV_SCP.$k";
		my $noisy_wav_scp_cur = "$CONFIG_DIR/N$i\_SNR$snrs[$j].scp.noisy.$k";

		my $NOISY_WAV_DIR = "positive_addnoise/SNR";
		$NOISY_WAV_DIR = "$NOISY_WAV_DIR\_$snrs[$j]db";

		system("mkdir -p $NOISY_WAV_DIR/N$k\_SNR$snrs[$j]");
		
		open(FILE_IN, "$clean_wav_scp_cur");
		open(FILE_OUT1, ">$noisy_wav_scp_cur");
		
		while(<FILE_IN>)
		{
			chomp;
			s#${CLEAN_WAV_DIR}#${NOISY_WAV_DIR}#;
			print FILE_OUT1 "$_\n";
		}
		close(FILE_IN);
		close(FILE_OUT1);
		
		MakePathForScp("$noisy_wav_scp_cur");
		
		system("$BIN_DIR/AddNoise_MultiOutput -i $clean_wav_scp_cur -o $noisy_wav_scp_cur -n $noise_file -s $snrs[$j] -r 1000 -multiple $output_type -m snr_8khz -u -d -e $ROOT_DIR/log/N$i\_SNR$snrs[$j].log");
	
	}
}

sub ::SplitFile
{
	my ($file, $n, $suffix, $separator) = @_;
	my ($i, $j, $k, $file_iii);
	my $line;
	my $numPerSplit;
	my $numLeft;
	
	my @lines;
	my @numPerSplit;
	
	$suffix = "" if(!defined($suffix));
	$separator = "." if(!defined($separator));
	
	open(IN, "$file") || die "Error: Can't read file: $file, $!";
	@lines = <IN>;
	close(IN);
	$i = @lines;
	
	$numPerSplit = int($i/$n);
	$numLeft = $i%$n;
	die "Error: Number of lines per split is 0, Total: $i, nSplit: $n\n" if($numPerSplit == 0);
	
	$numPerSplit[0] = $i;
	foreach(1..$n)
	{
		$numPerSplit[$_] = $numPerSplit;
	}
	foreach(1..$numLeft)
	{
		$numPerSplit[$_]++;
	}
	print "Number of lines for each split(SplitFile): @numPerSplit\n";
	
	foreach $k(1..$n)
	{
		$file_iii = $file.$separator.$k.$suffix;
		open(OUT, ">$file_iii") || die "Error: Can't write file: $file_iii, $!";
		foreach $j(1..$numPerSplit[$k])
		{
			print OUT shift(@lines);
		}
		close(OUT);
	}
}

sub ::MakePathForScp
{
	use File::Basename;
	
	my ($scp) = @_;
	open(IN, $scp) || die "Can't open file: $scp\n";
	while(<IN>)
	{
		chomp;
		my($strFilename, $strPathname, $suffix) = fileparse($_);
		::MakePathIfNotExist($strPathname);
	}
	close(IN);
}

sub ::MakePathIfNotExist
{
	use File::Path;
	my ($strPathname) = @_;
	
	if(!-e $strPathname)
	{
		mkpath($strPathname, 1, 0755) || die "Error: Can't make path: $strPathname, $!";
	}
}

sub ::MakeDirIfNotExist
{
	my ($strPathname) = @_;
	
	if(!-e $strPathname)
	{
		mkdir($strPathname, 0755) || die "Error: Can't make directory: $strPathname, $!";
	}
}

sub ::ExistOrWait
{
	@_ >=1 && @_ <= 2 || die "Error(ExistOrWait): Invalid parameter count, Usage: ExistOrWait(\$file, \$time]).";
	my ($file, $time) = @_;
	$time = 30 if(@_ < 2);
	while (!-e $file)
	{
		print ("Waiting $file $time minutes!\n");
		sleep(60*$time);
	}
}

sub ::ExistOrDie
{
	@_ >= 1 && @_ <= 2 || die "Error(ExistOrDie): Invalid parameter count, Usage: ExistOrDie(\$file, \$type).";
	-e $_[0] || die "Error: Missing or Can't open $_[1] file: $_[0].";
}